import torch

from torch import nn
from torch.utils.data import default_collate

from evlearn.nn.layers.conv_lstm import ConvLSTMCellStack
from evlearn.bundled.rtdetr_pytorch.zoo.rtdetr.hypergraph import SoftHGNNwithFFTMax

"""
多尺度特征的超图lstm
"""

def reset_by_mask(values, mask):
    # values : (N, ...)
    # mask   : (N, )

    mask = (~mask).to(values.dtype)
    mask = mask.reshape(mask.shape + (1,) * (values.ndim - 1))

    return mask * values


class TemporalConvLSTM(nn.Module):

    def __init__(
        self, fpn_shapes, n_layers_list,
        hidden_features_list = None, kernel_size_list = 3,
        rezero = True, hyper_edge_num = 16,
    ):
        # pylint: disable=too-many-arguments
        super().__init__()
        assert len(n_layers_list) == len(fpn_shapes)

        if isinstance(kernel_size_list, int):
            kernel_size_list = [kernel_size_list,] * len(fpn_shapes)

        if hidden_features_list is None:
            hidden_features_list = [shape[0] for shape in fpn_shapes]
        elif isinstance(hidden_features_list, int):
            hidden_features_list = [hidden_features_list,] * len(fpn_shapes)

        layers = []
        proj_layers = []

        self._hidden_features_list = hidden_features_list

        for (fpn_shape, n_layers, hidden_features, ks) in zip(
            fpn_shapes, n_layers_list, hidden_features_list, kernel_size_list
        ):
            if (n_layers is None) or (n_layers == 0):
                layers.append(None)
                proj_layers.append(None)
                continue

            layers.append(ConvLSTMCellStack(
                n_layers, fpn_shape[0], hidden_features, ks
            ))
            proj_layers.append(
                nn.Conv2d(hidden_features, fpn_shape[0], kernel_size=1)
            )

        self.layers      = nn.ModuleList(layers)
        self.proj_layers = nn.ModuleList(proj_layers)

        self._rezero = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1,)))
        else:
            self.re_alpha = 1.0

        # -----------------------------
        # 跨尺度特征交互（3 层 FPN 时启用）
        # -----------------------------
        self._use_cross_scale = (len(fpn_shapes) == 3)
        if self._use_cross_scale:
            c0 = fpn_shapes[0][0]
            c1 = fpn_shapes[1][0]
            c2 = fpn_shapes[2][0]
            assert c0 == c1 == c2, "cross-scale fusion assumes same channels per level"
            c = c0
            if c % hyper_edge_num != 0:
                raise ValueError(
                    f"fpn channel count ({c}) must be divisible by "
                    f"hyper_edge_num ({hyper_edge_num})"
                )

            # 上下采样模块（假设相邻层 H/W 比例为 2）
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
            self.upsample   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

            self.ln = nn.LayerNorm(c)
            self.hyper = SoftHGNNwithFFTMax(
                c=c, hyper_edge_num=hyper_edge_num, dropout=0.1
            )

            # 在中间尺度上，将三尺度特征 cat 后，1x1 Conv 融合为 C 通道
            self.cs_fuse_mid = nn.Conv2d(c * 3, c, kernel_size=1)

            # 将融合后的信息再回流到三层（Residual 形式）
            self.cs_refine0 = nn.Conv2d(c * 2, c, kernel_size=1)
            self.cs_refine0 = nn.Conv2d(c * 2, c, kernel_size=1)
            self.cs_refine1 = nn.Conv2d(c * 2, c, kernel_size=1)
            self.cs_refine2 = nn.Conv2d(c * 2, c, kernel_size=1)

            if rezero:
                self.cs_alpha = nn.Parameter(torch.zeros((1,)))
                # ReZero 风格门控，初始为 0，稳一点
                self.cs_alpha = nn.Parameter(torch.zeros((1,)))
            else:
                self.cs_alpha = 1.0
        else:
            self.cs_alpha = None

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha,)

    def init_mem(self, fpn_features_list):
        result = []

        for (fpn_features, layer) in zip(fpn_features_list, self.layers):
            if layer is None:
                result.append(None)
            else:
                result.append(layer.init_states(fpn_features))

        return result

    def reset_mem_by_mask(self, memory, reset_mask):
        result = []

        for layer_memory in memory:
            if layer_memory is None:
                result.append(None)
            else:
                result.append([
                    {
                        k: reset_by_mask(v, reset_mask)
                        for (k, v) in lstm_layer.items()
                    }
                    for lstm_layer in layer_memory
                ])

        return result

    def slice_mem(self, memory, batch_index):
        result = []

        for layer_memory in memory:
            if layer_memory is None:
                result.append(None)
            else:
                result.append([
                    {k: v[batch_index] for (k, v) in lstm_layer.items()}
                    for lstm_layer in layer_memory
                ])

        return result

    def detach_mem(self, memory):
        result = []

        for layer_memory in memory:
            if layer_memory is None:
                result.append(None)
            else:
                result.append([
                    {k: v.detach() for (k, v) in lstm_layer.items()}
                    for lstm_layer in layer_memory
                ])

        return result

    def collate_mem(self, memory_list):
        if not memory_list:
            return None

        result = []

        for layer_idx, layer in enumerate(self.layers):
            if layer is None:
                result.append(None)
            else:
                result.append(default_collate(
                    [x[layer_idx] for x in memory_list]
                ))

        return result

    def forward(self, fpn_features_list, memory):
        # fpn_features_list : List[ (N, C, H, W) ]
        # memory            : List[ (N, F, H, W) ]
        result     = []
        new_memory = []

        # 1) 各尺度独立 ConvLSTM（和原始版本一样）
        for (fpn_features, layer_memory, layer, proj_layer) in zip(
            fpn_features_list, memory, self.layers, self.proj_layers
        ):
            if layer is None:
                result.append(fpn_features)
                new_memory.append(None)
            else:
                lstm_encoding, new_layer_memory = layer(fpn_features, layer_memory)
                lstm_encoding = proj_layer(lstm_encoding)

                new_fpn_features = fpn_features + self.re_alpha * lstm_encoding

                result.append(new_fpn_features)
                new_memory.append(new_layer_memory)
            lm = memory[0]          # 第一个 FPN level
            st0 = lm[0]             # 第0层 ConvLSTMCell 的 state dict
            # print(st0.keys())
            # print({k: v.shape for k, v in st0.items() if torch.is_tensor(v)})

        # 2) 可选：对输出特征做一次跨尺度交互（不改 memory，只改 result）
        if self._use_cross_scale:
            # 假设 3 个尺度从高分辨率到低分辨率：[P2, P3, P4] 之类
            f0, f1, f2 = result  # shapes: (N, C, H0,W0), (N,C,H1,W1), (N,C,H2,W2)

            # 对齐到中间尺度 H1 x W1
            f0_d = self.downsample(f0)  # H0,W0 -> H1,W1
            f2_u = self.upsample(f2)    # H2,W2 -> H1,W1

            # 在中间尺度上融合三尺度信息
            mid_cat   = torch.cat([f0_d, f1, f2_u], dim=1)  # [N, 3C, H1, W1]
            mid_fused = self.cs_fuse_mid(mid_cat)           # [N, C,  H1, W1]

            hyper_encoding = mid_fused
            b, c, h, w = hyper_encoding.shape[0], hyper_encoding.shape[1], hyper_encoding.shape[2], hyper_encoding.shape[3]
            hyper_encoding = hyper_encoding.view(b, c, -1).contiguous().transpose(1, 2).contiguous()
            hyper_encoding = self.ln(hyper_encoding)
            hyper_encoding = self.hyper(hyper_encoding)
            hyper_encoding = hyper_encoding.transpose(1, 2).contiguous().view(b, c, h, w).contiguous()

            # mid_fused = hyper_encoding
            # mid_fused = hyper_encoding + mid_fused
            mid_fused = hyper_encoding

            # 再把 mid_fused 回流到三层（Residual）
            f0_u = self.upsample(mid_fused)   # -> H0,W0
            f2_d = self.downsample(mid_fused) # -> H2,W2

            delta0 = self.cs_refine0(torch.cat([f0, f0_u], dim=1))
            delta1 = self.cs_refine1(torch.cat([f1, mid_fused], dim=1))
            delta2 = self.cs_refine2(torch.cat([f2, f2_d], dim=1))

            alpha = self.cs_alpha if self.cs_alpha is not None else 1.0

            f0 = f0 + alpha * delta0
            f1 = f1 + alpha * delta1
            f2 = f2 + alpha * delta2

            result = [f0, f1, f2]

        return (result, new_memory)
