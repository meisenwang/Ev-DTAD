import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_uniform_
from .utils import get_activation

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SoftHGNNbw(nn.Module):
    def __init__(self, c, d_k=32, hyper_edge_num=4, dropout=0.1):
        super().__init__()
        self.edge_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.hyper_edge_num = hyper_edge_num
        self.d_k = d_k
        self.group_dim = c // hyper_edge_num
        self.prototype = nn.Parameter(torch.Tensor(hyper_edge_num, d_k))
        nn.init.xavier_uniform_(self.prototype)
        self.token_q_proj = nn.Linear(c, d_k)
        self.token_k_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.group_dim * 2, d_k),
            nn.Unflatten(1, (d_k, 1)))
        self.gate = nn.Parameter(torch.ones(c)) # torch.Tensor(num_node)
        self.c = c
        nn.init.constant_(self.gate, 1)

    def forward(self, x_in):
        x = x_in
        Q = self.token_q_proj(x)
        avg_context = x.mean(dim=1, keepdim=True)
        max_context, _ = x.max(dim=1, keepdim=True)
        context_cat = torch.cat([avg_context, max_context], dim=1)
        x_split = torch.chunk(context_cat, self.hyper_edge_num, dim=2)
        x_split = torch.cat(x_split, dim=0)
        outputs = self.token_k_proj(x_split)
        outputs = torch.chunk(outputs.squeeze(2), self.hyper_edge_num, dim=0)
        K = torch.stack(outputs, dim=1) + self.prototype.unsqueeze(0)
        hg = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)
        hg = torch.nan_to_num(hg, 0)
        hg = self.dropout(hg)
        hg = F.softmax(hg, dim=1)
        He = torch.bmm(hg.transpose(1, 2), x)
        He = self.edge_proj(He)
        X_new = torch.bmm(hg, He)
        X_new = self.node_proj(X_new)
        return X_new * self.gate.unsqueeze(0).unsqueeze(0)

class OffsetGen(nn.Module):
    def __init__(
            self,
            embed_dims: int = 256,
            num_heads: int = 8,
            num_edges: int = 16):
        super(OffsetGen, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.head_dim = embed_dims // num_heads

        self.in_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(self.head_dim, self.head_dim * num_edges)

    def forward(self, x):
        x = self.in_proj(x.mean(1))
        x = x.reshape(x.shape[0], self.num_heads, self.head_dim)
        x = self.act(x)
        x = self.out_proj(x)
        x = x.reshape(x.shape[0], self.num_heads, self.num_edges, self.head_dim)
        return x


class MultiheadHyperGraph(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 num_edges: int = 16,
                 attn_drop: float = 0):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.attn_drop = attn_drop

        self.gen_offset = OffsetGen(embed_dims, num_heads, num_edges)
        self.prototype = nn.Parameter(torch.randn(num_edges, embed_dims))
        self.in_proj_vertex_k = nn.Linear(embed_dims, embed_dims)
        self.in_proj_vertex_v = nn.Linear(embed_dims, embed_dims)
        self.out_proj_edge = nn.Linear(embed_dims, embed_dims)
        self.out_proj_edge = nn.Linear(embed_dims, embed_dims)
        self.out_proj_vertex = nn.Linear(embed_dims, embed_dims)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.prototype)
        xavier_uniform_(self.in_proj_vertex_k.weight)
        xavier_uniform_(self.in_proj_vertex_v.weight)
        constant_(self.in_proj_vertex_k.bias, 0.)
        constant_(self.in_proj_vertex_v.bias, 0.)
        constant_(self.out_proj_edge.bias, 0.)
        constant_(self.out_proj_vertex.bias, 0.)

    def _in_shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return tensor.reshape(
            bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).reshape(
            bsz * self.num_heads, seq_len, self.head_dim)

    def _out_shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return tensor.reshape(
            bsz, self.num_heads, seq_len, self.head_dim).transpose(1, 2).reshape(
            bsz, seq_len, self.num_heads * self.head_dim)

    def forward(
            self,
            vertex: Tensor,):
        bsz, vertex_len, embed_dims = vertex.shape
        offset = self.gen_offset(vertex)
        proto = self.prototype.reshape(self.num_edges, self.num_heads, self.head_dim).permute(1, 0, 2)
        edge = offset + proto.unsqueeze(0)
        edge = edge.reshape(-1, self.num_edges, self.head_dim) * self.scale
        vertex_k = self._in_shape(self.in_proj_vertex_k(vertex), vertex_len, bsz)
        vertex_v = self._in_shape(self.in_proj_vertex_v(vertex), vertex_len, bsz)

        attn_weights_edge_vertex = torch.bmm(edge, vertex_k.transpose(-2, -1))
        if self.attn_drop > 0.0:
            attn_weights_edge_vertex = F.dropout(attn_weights_edge_vertex, p=self.attn_drop)
        attn_weights_vertex_edge = attn_weights_edge_vertex.transpose(-2, -1)
        attn_weights_edge_vertex = F.softmax(attn_weights_edge_vertex, dim=-1)
        attn_weights_vertex_edge = F.softmax(attn_weights_vertex_edge, dim=-1)

        output_edge = self._out_shape(
            torch.bmm(attn_weights_edge_vertex, vertex_v), self.num_edges, bsz)
        output_edge = self.out_proj_edge(output_edge)
        output_edge = self._in_shape(output_edge, self.num_edges, bsz)

        output_vertex = self._out_shape(
            torch.bmm(attn_weights_vertex_edge, output_edge), vertex_len, bsz)
        output_vertex = self.out_proj_vertex(output_vertex)
        return output_vertex

class HyperFormer(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 num_edges: int = 16,
                 attn_drop: float = 0,
                 dropout: float = 0.,
                 ffn_ratio: int = 4):
        super().__init__()

        self.graph = MultiheadHyperGraph(embed_dims, num_heads, num_edges, attn_drop)

        self.linear1 = nn.Linear(embed_dims, embed_dims * ffn_ratio)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dims * ffn_ratio, embed_dims)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src) -> torch.Tensor:
        b, c, h, w = src.shape
        src = src.flatten(2).permute(0, 2, 1)
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c, 10000.0).to(src.device)
        src = self.with_pos_embed(src, pos_embed)

        residual = src
        src = self.graph(src)
        src = residual + self.dropout1(src)
        src = self.norm1(src)

        residual = src
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        src = self.norm2(src)
        src = src.permute(0, 2, 1).reshape(-1, c, h, w).contiguous()
        return src

class DualHyperFormer(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 num_edges: int = 16,
                 attn_drop: float = 0,
                 dropout: float = 0.,
                 ffn_ratio: int = 4):
        super().__init__()

        self.m1 = HyperFormer(embed_dims, num_heads, num_edges, attn_drop, dropout, ffn_ratio)
        self.m2 = HyperFormer(embed_dims, num_heads, num_edges, attn_drop, dropout, ffn_ratio)

    def forward(self, x1, x2):
        out1_ = torch.chunk(x1, 4, dim=1)
        out2_ = torch.chunk(x2, 4, dim=1)
        out1 = torch.cat([out1_[0], out2_[1], out1_[2], out2_[3]], dim=1)
        out2 = torch.cat([out2_[0], out1_[1], out2_[2], out1_[3]], dim=1)
        out1 = self.m1(out1) + x1
        out2 = self.m2(out2) + x2
        return out1, out2
    
class SoftHGNN(nn.Module):
    def __init__(self, c, d_k=32, hyper_edge_num=16, dropout=0.1, num_node=320):
        super().__init__()

        self.edge_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())

        self.dropout = nn.Dropout(dropout)
        self.hyper_edge_num = hyper_edge_num
        self.d_k = d_k
        self.group_dim = c // hyper_edge_num

        self.prototype = nn.Parameter(torch.Tensor(hyper_edge_num, d_k))
        nn.init.xavier_uniform_(self.prototype)

        self.token_q_proj = nn.Linear(c, d_k)
        # self.token_k_proj = nn.Linear(self.group_dim, d_k)
        self.token_k_proj = nn.Sequential(nn.Flatten(), nn.Linear(self.group_dim * 2, d_k), nn.Unflatten(1, (d_k, 1)))
        self.gate = nn.Parameter(torch.ones(num_node))

    def forward(self, x_in):
        x = x_in

        # b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # x = x.view(b, c, -1).transpose(1, 2).contiguous()
        b, n, c = x.shape

        Q = self.token_q_proj(x)  # 256 -> 32

        avg_context = x.mean(dim=1, keepdim=True)
        max_context, _ = x.max(dim=1, keepdim=True)
        context_cat = torch.cat([avg_context, max_context], dim=1)   # 2, 256

        # x_split = context_cat.reshape(b, 2, self.hyper_edge_num//2, self.group_dim).permute(0, 2, 1, 3).reshape(-1,2,self.group_dim)
        # outputs = self.token_k_proj(x_split)
        # outputs = outputs.reshape(b, self.hyper_edge_num, -1).permute(0, 2, 1)
        # K = outputs + self.prototype

        x_split = torch.chunk(context_cat, self.hyper_edge_num, dim=2)  # hyper_edge_num个 bs,N,fea/hyper_edge_num
        x_split = torch.cat(x_split, dim=0)

        outputs = self.token_k_proj(x_split)
        outputs = torch.chunk(outputs.squeeze(2), self.hyper_edge_num, dim=0)
        K = torch.stack(outputs, dim=1) + self.prototype.unsqueeze(0)

        hg = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(self.d_k)
        hg = F.softmax(hg, dim=1)
        hg = self.dropout(hg)


        He = torch.bmm(hg.transpose(1, 2), x)
        He = self.edge_proj(He)
        X_new = torch.bmm(hg, He)
        X_new = self.node_proj(X_new)

        # X_new = X_new.transpose(1, 2).contiguous().view(b, c, h, w)

        return x_in + X_new*self.gate.unsqueeze(1).unsqueeze(0)
    
class SoftHGNN50(nn.Module):
    def __init__(self, c, d_k=32, hyper_edge_num=16, dropout=0.1, num_node=320):
        super().__init__()

        self.edge_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())

        self.dropout = nn.Dropout(dropout)
        self.hyper_edge_num = hyper_edge_num
        self.d_k = d_k
        self.group_dim = c // hyper_edge_num

        self.prototype = nn.Parameter(torch.Tensor(hyper_edge_num, d_k))
        nn.init.xavier_uniform_(self.prototype)

        self.token_q_proj = nn.Linear(c, d_k)
        # self.token_k_proj = nn.Linear(self.group_dim, d_k)
        self.token_k_proj = nn.Sequential(nn.Flatten(), nn.Linear(self.group_dim * 2, d_k), nn.Unflatten(1, (d_k, 1)))
        self.gate = nn.Parameter(torch.zeros(num_node))

    def forward(self, x_in):
        x = x_in

        # b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # x = x.view(b, c, -1).transpose(1, 2).contiguous()
        b, n, c = x.shape

        Q = self.token_q_proj(x)  # 256 -> 32

        avg_context = x.mean(dim=1, keepdim=True)
        max_context, _ = x.max(dim=1, keepdim=True)
        context_cat = torch.cat([avg_context, max_context], dim=1)   # 2, 256

        # x_split = context_cat.reshape(b, 2, self.hyper_edge_num//2, self.group_dim).permute(0, 2, 1, 3).reshape(-1,2,self.group_dim)
        # outputs = self.token_k_proj(x_split)
        # outputs = outputs.reshape(b, self.hyper_edge_num, -1).permute(0, 2, 1)
        # K = outputs + self.prototype

        x_split = torch.chunk(context_cat, self.hyper_edge_num, dim=2)  # hyper_edge_num个 bs,N,fea/hyper_edge_num
        x_split = torch.cat(x_split, dim=0)

        outputs = self.token_k_proj(x_split)
        outputs = torch.chunk(outputs.squeeze(2), self.hyper_edge_num, dim=0)
        K = torch.stack(outputs, dim=1) + self.prototype.unsqueeze(0)

        hg = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(self.d_k)
        hg = F.softmax(hg, dim=2)
        hg = self.dropout(hg)


        He = torch.bmm(hg.transpose(1, 2), x)
        He = self.edge_proj(He)
        X_new = torch.bmm(hg, He)
        X_new = self.node_proj(X_new)

        # X_new = X_new.transpose(1, 2).contiguous().view(b, c, h, w)

        return x_in + X_new*self.gate.unsqueeze(1).unsqueeze(0)

class SoftHGNN2(nn.Module):
    def __init__(self, c, d_k=32, hyper_edge_num=16, dropout=0.1, num_node=65):
        super().__init__()
        self.edge_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.hyper_edge_num = hyper_edge_num
        self.d_k = d_k
        self.group_dim = c // hyper_edge_num
        self.prototype = nn.Parameter(torch.Tensor(hyper_edge_num, d_k))
        nn.init.xavier_uniform_(self.prototype)
        self.token_q_proj = nn.Linear(c, d_k)
        self.token_k_proj = nn.Sequential(nn.Flatten(), nn.Linear(self.group_dim * 2, d_k), nn.Unflatten(1, (d_k, 1)))
        self.c = c
        self.gate = nn.Parameter(torch.ones(c)) # torch.Tensor(num_node)
        # nn.init.constant_(self.gate, 1.0)

    def forward(self, x_in):
        # print("X_in.shape:", x_in.shape)
        # print("self.c: ", self.c)
        x = x_in
        b, n, c = x.shape
        Q = self.token_q_proj(x)
        avg_context = x.mean(dim=1, keepdim=True)
        max_context, _ = x.max(dim=1, keepdim=True)
        context_cat = torch.cat([avg_context, max_context], dim=1)
        x_split = torch.chunk(context_cat, self.hyper_edge_num, dim=2)
        x_split = torch.cat(x_split, dim=0)
        outputs = self.token_k_proj(x_split)
        outputs = torch.chunk(outputs.squeeze(2), self.hyper_edge_num, dim=0)
        K = torch.stack(outputs, dim=1) + self.prototype.unsqueeze(0)
        hg = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)
        hg = self.dropout(hg)
        hg = F.softmax(hg, dim=1)
        He = torch.bmm(hg.transpose(1, 2), x)
        He = self.edge_proj(He)
        X_new = torch.bmm(hg, He)
        X_new = self.node_proj(X_new)
        # print("X_new.shape:", X_new.shape)
        # print("x_in.shape:", x_in.shape)
        # print("self.gate.shape:", self.gate.shape)
        # return x_in + X_new # self.gate.unsqueeze(1).unsqueeze(0)
        return x_in + X_new * self.gate.unsqueeze(0).unsqueeze(-0) # self.gate.unsqueeze(1).unsqueeze(0)

class SoftHGNNwithFFT(nn.Module):
    def __init__(self, c, d_k=32, hyper_edge_num=16, dropout=0.1, num_node=65):
        super().__init__()
        self.num_context = 3  # avg, max, freq  👈 新增
        self.edge_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.hyper_edge_num = hyper_edge_num
        self.d_k = d_k
        self.group_dim = c // hyper_edge_num
        self.prototype = nn.Parameter(torch.Tensor(hyper_edge_num, d_k))
        nn.init.xavier_uniform_(self.prototype)
        self.token_q_proj = nn.Linear(c, d_k)
        self.token_k_proj = nn.Sequential(nn.Flatten(), nn.Linear(self.group_dim * self.num_context, d_k), nn.Unflatten(1, (d_k, 1)))
        self.c = c
        self.gate = nn.Parameter(torch.ones(c)) # torch.Tensor(num_node)
        # nn.init.constant_(self.gate, 1.0)

    def forward(self, x_in):
        x = x_in                               # (B,N,C)
        b, n, c = x.shape

        # -------- Q: 节点查询 --------
        Q = self.token_q_proj(x)              # (B,N,d_k)

        # -------- 空间域全局上下文 --------
        avg_context = x.mean(dim=1, keepdim=True)            # (B,1,C)
        max_context, _ = x.max(dim=1, keepdim=True)          # (B,1,C)

        # -------- 频域全局上下文（关键）--------
        # 在 token 维度 N 上做 1D FFT，得到每个通道的频谱
        freq = torch.fft.rfft(x, dim=1)                      # (B, N_fft, C), complex
        # 用幅值的平均作为简单的“频率能量”统计
        freq_energy = freq.abs().mean(dim=1, keepdim=True)   # (B,1,C)

        # 拼成多模态上下文: 空间 avg + 空间 max + 频域能量
        context_cat = torch.cat(
            [avg_context, max_context, freq_energy], dim=1
        )                                                    # (B,3,C)

        # -------- 通道分组构造每条超边的上下文 --------
        # 在 C 维上均分成 H 组
        x_split = torch.chunk(context_cat, self.hyper_edge_num, dim=2)
        # x_split 是长度 H 的 tuple，每个 (B,3,group_dim)
        x_split = torch.cat(x_split, dim=0)                  # (B*H,3,group_dim)

        # -------- 生成 K（频域增强的超边原型）--------
        outputs = self.token_k_proj(x_split)                 # (B*H,d_k,1)
        outputs = torch.chunk(outputs.squeeze(2), self.hyper_edge_num, dim=0)
        K = torch.stack(outputs, dim=1)                      # (B,H,d_k)
        K = K + self.prototype.unsqueeze(0)                  # (B,H,d_k)

        # -------- 软超图关联矩阵 hg --------
        hg = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)  # (B,N,H)
        hg = self.dropout(hg)
        hg = F.softmax(hg, dim=1)                            # 对节点维 N softmax

        # -------- node -> hyper-edge --------
        He = torch.bmm(hg.transpose(1, 2), x)                # (B,H,C)
        He = self.edge_proj(He)                              # (B,H,C)

        # -------- hyper-edge -> node --------
        X_new = torch.bmm(hg, He)                            # (B,N,C)
        X_new = self.node_proj(X_new)                        # (B,N,C)

        # -------- 通道 gate + 残差 --------
        return x_in + X_new * self.gate.view(1, 1, -1)
    # self.gate.unsqueeze(1).unsqueeze(0)

class SoftHGNNwithFFTMax(nn.Module):
    def __init__(self, c, d_k=32, hyper_edge_num=16, dropout=0.1, num_node=65):
        super().__init__()
        self.num_context = 4  # avg, max, freq  👈 新增
        self.edge_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.hyper_edge_num = hyper_edge_num
        self.d_k = d_k
        self.group_dim = c // hyper_edge_num
        self.prototype = nn.Parameter(torch.Tensor(hyper_edge_num, d_k))
        nn.init.xavier_uniform_(self.prototype)
        self.token_q_proj = nn.Linear(c, d_k)
        self.token_k_proj = nn.Sequential(nn.Flatten(), nn.Linear(self.group_dim * self.num_context, d_k), nn.Unflatten(1, (d_k, 1)))
        self.c = c
        self.gate = nn.Parameter(torch.zeros(c)) # torch.Tensor(num_node)
        # nn.init.constant_(self.gate, 1.0)

    def forward(self, x_in):
        x = x_in                               # (B,N,C)
        b, n, c = x.shape

        # -------- Q: 节点查询 --------
        Q = self.token_q_proj(x)              # (B,N,d_k)

        # -------- 空间域全局上下文 --------
        avg_context = x.mean(dim=1, keepdim=True)            # (B,1,C)
        max_context, _ = x.max(dim=1, keepdim=True)          # (B,1,C)

        # -------- 频域全局上下文（关键）--------
        # 在 token 维度 N 上做 1D FFT，得到每个通道的频谱
        freq = torch.fft.rfft(x, dim=1)                      # (B, N_fft, C), complex
        # 用幅值的平均作为简单的“频率能量”统计
        freq_energy_abs = freq.abs().mean(dim=1, keepdim=True)   # (B,1,C)
        freq_energy_max, _ = freq.abs().max(dim=1, keepdim=True)

        # 拼成多模态上下文: 空间 avg + 空间 max + 频域能量
        context_cat = torch.cat(
            [avg_context, max_context, freq_energy_max, freq_energy_abs], dim=1
        )                                                    # (B,3,C)

        # -------- 通道分组构造每条超边的上下文 --------
        # 在 C 维上均分成 H 组
        x_split = torch.chunk(context_cat, self.hyper_edge_num, dim=2)
        # x_split 是长度 H 的 tuple，每个 (B,3,group_dim)
        x_split = torch.cat(x_split, dim=0)                  # (B*H,3,group_dim)

        # -------- 生成 K（频域增强的超边原型）--------
        outputs = self.token_k_proj(x_split)                 # (B*H,d_k,1)
        outputs = torch.chunk(outputs.squeeze(2), self.hyper_edge_num, dim=0)
        K = torch.stack(outputs, dim=1)                      # (B,H,d_k)
        K = K + self.prototype.unsqueeze(0)                  # (B,H,d_k)

        # -------- 软超图关联矩阵 hg --------
        hg = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)  # (B,N,H)
        hg = self.dropout(hg)
        hg = F.softmax(hg, dim=1)                            # 对节点维 N softmax

        # -------- node -> hyper-edge --------
        He = torch.bmm(hg.transpose(1, 2), x)                # (B,H,C)
        He = self.edge_proj(He)                              # (B,H,C)

        # -------- hyper-edge -> node --------
        X_new = torch.bmm(hg, He)                            # (B,N,C)
        X_new = self.node_proj(X_new)                        # (B,N,C)

        # -------- 通道 gate + 残差 --------
        return x_in + X_new * self.gate.view(1, 1, -1)
    # self.gate.unsqueeze(1).unsqueeze(0)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTHGNNwithMultihead(nn.Module):
    def __init__(self, c, d_k=128, hyper_edge_num=16, dropout=0.1, num_node=65, num_heads=4):
        super().__init__()
        self.num_context = 4  # avg, max, freq_max, freq_abs_mean

        self.c = c
        self.hyper_edge_num = hyper_edge_num  # H
        self.d_k = d_k
        self.num_heads = num_heads  # M

        assert d_k % num_heads == 0, f"d_k({d_k}) must be divisible by num_heads({num_heads})"
        self.d_head = d_k // num_heads

        assert c % hyper_edge_num == 0, f"c({c}) must be divisible by hyper_edge_num({hyper_edge_num})"
        self.group_dim = c // hyper_edge_num

        self.dropout = nn.Dropout(dropout)

        # ---- Q / V 投影（多头通过 reshape 切分）----
        self.token_q_proj = nn.Linear(c, d_k)
        self.token_v_proj = nn.Linear(c, d_k)

        # ---- 从每条超边的上下文生成 K（总维度 d_k）----
        self.token_k_proj = nn.Sequential(
            nn.Flatten(1),  # (B*H, num_context, group_dim) -> (B*H, num_context*group_dim)
            nn.Linear(self.group_dim * self.num_context, d_k),
        )

        # ✅ 多头化 prototype：每条超边、每个 head 各自一套原型
        # shape: (H, M, d_head)
        self.prototype = nn.Parameter(torch.empty(hyper_edge_num, num_heads, self.d_head))
        nn.init.xavier_uniform_(self.prototype)

        # ---- 超边特征处理（在 d_k 空间做）----
        self.edge_proj = nn.Sequential(nn.Linear(d_k, d_k), nn.GELU())

        # ---- 输出回到 C，并保留你原来的 node_proj ----
        self.out_proj = nn.Linear(d_k, c)
        self.node_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())

        # ---- 通道 gate + 残差 ----
        self.gate = nn.Parameter(torch.zeros(c))

    def forward(self, x_in):
        """
        x_in: (B, N, C)
        """
        x = x_in
        B, N, C = x.shape
        H = self.hyper_edge_num
        M = self.num_heads
        Dh = self.d_head

        # -------- Multi-Head Q / V --------
        Q = self.token_q_proj(x)  # (B, N, d_k)
        V = self.token_v_proj(x)  # (B, N, d_k)

        # -> (B, M, N, Dh)
        Q = Q.view(B, N, M, Dh).permute(0, 2, 1, 3).contiguous()
        V = V.view(B, N, M, Dh).permute(0, 2, 1, 3).contiguous()

        # -------- 空间域上下文 --------
        avg_context = x.mean(dim=1, keepdim=True)        # (B, 1, C)
        max_context, _ = x.max(dim=1, keepdim=True)      # (B, 1, C)

        # -------- 频域上下文 --------
        freq = torch.fft.rfft(x, dim=1)                  # (B, N_fft, C), complex
        freq_abs = freq.abs()
        freq_energy_abs = freq_abs.mean(dim=1, keepdim=True)       # (B, 1, C)
        freq_energy_max, _ = freq_abs.max(dim=1, keepdim=True)     # (B, 1, C)

        # (B, 4, C)
        context_cat = torch.cat([avg_context, max_context, freq_energy_max, freq_energy_abs], dim=1)

        # -------- 通道分组：给每条超边构造上下文 --------
        # -> (B*H, 4, group_dim)
        x_split = torch.chunk(context_cat, H, dim=2)
        x_split = torch.cat(x_split, dim=0)

        # -------- 生成 K_total: (B, H, d_k) --------
        K_total = self.token_k_proj(x_split)             # (B*H, d_k)
        # 还原回 (B, H, d_k)，注意 chunk/cat 的顺序：先按 H 拼在 batch 上
        K_total = K_total.view(H, B, self.d_k).permute(1, 0, 2).contiguous()  # (B, H, d_k)

        # -------- reshape 成多头 K: (B, M, H, Dh) --------
        K = K_total.view(B, H, M, Dh).permute(0, 2, 1, 3).contiguous()        # (B, M, H, Dh)

        # ✅ 加上多头化 prototype
        # prototype: (H, M, Dh) -> (M, H, Dh) -> (1, M, H, Dh)
        proto = self.prototype.permute(1, 0, 2).unsqueeze(0)
        K = K + proto

        # -------- 多头软超图关联矩阵 hg --------
        # (B, M, N, H)
        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Dh)
        attn_logits = self.dropout(attn_logits)

        # 你原版是对节点维 N softmax：每条超边从所有节点吸收信息
        hg = F.softmax(attn_logits, dim=2)  # (B, M, N, H)
        hg = self.dropout(hg)

        # -------- node -> hyper-edge（每个 head）--------
        # (B, M, H, N) @ (B, M, N, Dh) -> (B, M, H, Dh)
        He = torch.matmul(hg.transpose(-2, -1), V)

        # head concat -> edge_proj(d_k) -> 再拆回 head
        He_cat = He.permute(0, 2, 1, 3).contiguous().view(B, H, self.d_k)  # (B, H, d_k)
        He_cat = self.edge_proj(He_cat)                                    # (B, H, d_k)
        He = He_cat.view(B, H, M, Dh).permute(0, 2, 1, 3).contiguous()     # (B, M, H, Dh)

        # -------- hyper-edge -> node（每个 head）--------
        # (B, M, N, H) @ (B, M, H, Dh) -> (B, M, N, Dh)
        X_head = torch.matmul(hg, He)

        # concat heads -> (B, N, d_k) -> out_proj -> (B, N, C)
        X_cat = X_head.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_k)
        X_new = self.out_proj(X_cat)          # (B, N, C)
        X_new = self.node_proj(X_new)         # (B, N, C)

        return x_in + X_new * self.gate.view(1, 1, -1)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftHGNNwithEnergyFFT(nn.Module):
    def __init__(self, c, d_k=32, hyper_edge_num=16, dropout=0.1,
                 num_node=65, energy_ratio=0.8):
        super().__init__()
        self.num_context = 3  # avg, max, freq
        self.edge_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(c, c), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.hyper_edge_num = hyper_edge_num
        self.d_k = d_k
        self.group_dim = c // hyper_edge_num
        self.prototype = nn.Parameter(torch.Tensor(hyper_edge_num, d_k))
        nn.init.xavier_uniform_(self.prototype)
        self.token_q_proj = nn.Linear(c, d_k)
        self.token_k_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.group_dim * self.num_context, d_k),
            nn.Unflatten(1, (d_k, 1)),
        )
        self.c = c
        self.gate = nn.Parameter(torch.ones(c))  # 通道 gate
        self.energy_ratio = energy_ratio        # 保留的累计能量比例，比如 0.8

    def forward(self, x_in):
        x = x_in                               # (B,N,C)
        b, n, c = x.shape

        # -------- Q: 节点查询 --------
        Q = self.token_q_proj(x)              # (B,N,d_k)

        # -------- 空间域全局上下文 --------
        avg_context = x.mean(dim=1, keepdim=True)            # (B,1,C)
        max_context, _ = x.max(dim=1, keepdim=True)          # (B,1,C)

        # -------- 频域全局上下文（按能量自适应截断）--------
        # 在 token 维度 N 上做 1D FFT，得到每个通道的频谱
        freq = torch.fft.rfft(x, dim=1)                      # (B, N_fft, C), complex
        freq_abs = freq.abs()                                # (B, N_fft, C)

        B, N_fft, C = freq_abs.shape

        # (B, N_fft, C) -> (B, C, N_fft) -> (B*C, N_fft)
        freq_abs_flat = freq_abs.permute(0, 2, 1).reshape(-1, N_fft)  # (B*C, N_fft)

        # 对每个 (batch, channel) 的频谱能量做降序排序
        sorted_vals, _ = torch.sort(freq_abs_flat, dim=1, descending=True)  # (B*C, N_fft)

        # 计算累计能量占比
        total_energy = sorted_vals.sum(dim=1, keepdim=True) + 1e-8
        cumsum_energy = sorted_vals.cumsum(dim=1)
        energy_ratio = cumsum_energy / total_energy          # (B*C, N_fft)

        # 只保留累计能量 <= self.energy_ratio 的频率分量
        mask = energy_ratio <= self.energy_ratio             # (B*C, N_fft)
        num_keep = mask.sum(dim=1).clamp(min=1)              # 避免为 0

        # 取被保留频率分量的平均能量
        kept_energy_sum = (sorted_vals * mask).sum(dim=1)    # (B*C,)
        freq_energy_flat = kept_energy_sum / num_keep        # (B*C,)

        # reshape 回 (B, 1, C)
        freq_energy = freq_energy_flat.view(B, C).unsqueeze(1)  # (B,1,C)

        # 拼成多模态上下文: 空间 avg + 空间 max + 频域能量
        context_cat = torch.cat(
            [avg_context, max_context, freq_energy], dim=1
        )                                                    # (B,3,C)

        # -------- 通道分组构造每条超边的上下文 --------
        # 在 C 维上均分成 H 组
        x_split = torch.chunk(context_cat, self.hyper_edge_num, dim=2)
        # x_split 是长度 H 的 tuple，每个 (B,3,group_dim)
        x_split = torch.cat(x_split, dim=0)                  # (B*H,3,group_dim)

        # -------- 生成 K（频域增强的超边原型）--------
        outputs = self.token_k_proj(x_split)                 # (B*H,d_k,1)
        outputs = torch.chunk(outputs.squeeze(2), self.hyper_edge_num, dim=0)
        K = torch.stack(outputs, dim=1)                      # (B,H,d_k)
        K = K + self.prototype.unsqueeze(0)                  # (B,H,d_k)

        # -------- 软超图关联矩阵 hg --------
        hg = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)  # (B,N,H)
        hg = self.dropout(hg)
        hg = F.softmax(hg, dim=1)                            # 对节点维 N softmax

        # -------- node -> hyper-edge --------
        He = torch.bmm(hg.transpose(1, 2), x)                # (B,H,C)
        He = self.edge_proj(He)                              # (B,H,C)

        # -------- hyper-edge -> node --------
        X_new = torch.bmm(hg, He)                            # (B,N,C)
        X_new = self.node_proj(X_new)                        # (B,N,C)

        # -------- 通道 gate + 残差 --------
        return x_in + X_new * self.gate.view(1, 1, -1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FreqBandSoftHGNN(nn.Module):
    """
    频带划分的 Soft-HGNN 版本：
    - 输入:  feature map (B, C, H, W)
    - 输出:  同形状 (B, C, H, W)
    - 超边 = 频带(low/mid/high) × 通道分组(groups_per_band)
    """

    def __init__(self,
                 in_channels: int,
                 d_k: int = 32,
                 bands: int = 3,              # 低/中/高三段
                 groups_per_band: int = 8,    # 每个频带多少个通道组
                 dropout: float = 0.1):
        super().__init__()
        assert bands == 3, "当前实现假定 3 个频带: 低/中/高"
        assert in_channels % groups_per_band == 0, \
            "in_channels 必须能被 groups_per_band 整除"

        self.c = in_channels
        self.d_k = d_k
        self.bands = bands
        self.groups_per_band = groups_per_band
        self.hyper_edge_num = bands * groups_per_band
        self.group_dim = in_channels // groups_per_band

        # 节点/超边投影
        self.token_q_proj = nn.Linear(in_channels, d_k)
        self.edge_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU()
        )

        # 用频带+通道组的统计特征 -> K
        self.token_k_proj = nn.Linear(self.group_dim, d_k)

        # 每个 (频带, 通道组) 一条 prototype
        # shape: (bands, groups_per_band, d_k)
        self.prototype = nn.Parameter(
            torch.Tensor(bands, groups_per_band, d_k)
        )
        nn.init.xavier_uniform_(self.prototype)

        self.dropout = nn.Dropout(dropout)

        # 通道级 gate
        self.gate = nn.Parameter(torch.ones(in_channels))

    def _build_band_masks(self, H, W, device):
        """
        根据 H, W 构造 低/中/高 频带掩码
        使用 fftfreq 得到归一化频率，半径 r in [0, 0.5]
        """
        fy = torch.fft.fftfreq(H, device=device).view(1, 1, H, 1)  # (1,1,H,1)
        fx = torch.fft.fftfreq(W, device=device).view(1, 1, 1, W)  # (1,1,1,W)
        radius = torch.sqrt(fx**2 + fy**2)                         # (1,1,H,W), 0~0.5

        # 你可以根据经验调整这两个阈值
        t1, t2 = 0.08, 0.22

        low_mask  = (radius < t1)              # 低频
        mid_mask  = (radius >= t1) & (radius < t2)  # 中频
        high_mask = (radius >= t2)             # 高频

        # (bands, 1, 1, H, W) -> 之后广播到 (B, C, H, W)
        masks = torch.stack([low_mask, mid_mask, high_mask], dim=0)  # (3,1,1,H,W)
        return masks

    def forward(self, x_map: torch.Tensor) -> torch.Tensor:
        """
        x_map: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x_map.shape
        assert C == self.c

        # ---------- 1) 节点特征: flatten 成 (B, N, C) ----------
        N = H * W
        x = x_map.flatten(2).transpose(1, 2)      # (B, N, C)

        # Q: 节点查询
        Q = self.token_q_proj(x)                  # (B, N, d_k)

        # ---------- 2) 频域分析: 2D FFT + 频带能量 ----------
        # 2D FFT on spatial dims
        Xf = torch.fft.fft2(x_map, dim=(-2, -1))  # (B, C, H, W), complex
        mag = Xf.abs()                            # 幅度谱 (B, C, H, W)

        # 构造低/中/高频掩码
        band_masks = self._build_band_masks(H, W, x_map.device)  # (3,1,1,H,W)

        # 对每个频带，算每个通道的平均能量，得到 band_context[b] ∈ (B,1,C)
        band_contexts = []
        for b in range(self.bands):
            mask = band_masks[b]                  # (1,1,H,W)
            # 广播到 (B,C,H,W)，然后在空间上求平均
            energy = (mag * mask).mean(dim=(-2, -1))  # (B, C)
            band_context = energy.unsqueeze(1)        # (B, 1, C)
            band_contexts.append(band_context)

        # ---------- 3) 按 “频带 × 通道分组” 构造每条超边的 K ----------
        Ks = []  # 最终会拼成 (B, H_e, d_k)
        for b in range(self.bands):
            # band_contexts[b]: (B, 1, C)
            ctx_b = band_contexts[b]
            # 在通道维上切成 groups_per_band 组，每组 (B,1,group_dim)
            ctx_b_groups = torch.chunk(ctx_b, self.groups_per_band, dim=2)

            for g in range(self.groups_per_band): 
                # 当前 (频带b, 组g) 的通道上下文
                group_ctx = ctx_b_groups[g].squeeze(1)   # (B, group_dim)

                # 线性投影到 d_k 维 + 对应 prototype
                K_bg = self.token_k_proj(group_ctx)      # (B, d_k)
                K_bg = K_bg + self.prototype[b, g].unsqueeze(0)  # (B, d_k)

                Ks.append(K_bg)

        # 堆叠所有超边: H_e = bands * groups_per_band
        K = torch.stack(Ks, dim=1)               # (B, H_e, d_k)

        # ---------- 4) 软超图关联矩阵 hg ----------
        # Q: (B, N, d_k), K: (B, H_e, d_k)
        hg = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)  # (B, N, H_e)
        hg = self.dropout(hg)
        # 在节点维度 N 上 softmax: 每条超边在所有节点上的归一化权重
        hg = F.softmax(hg, dim=1)               # (B, N, H_e)

        # ---------- 5) node -> hyper-edge ----------
        # hg^T: (B, H_e, N), x: (B, N, C)
        He = torch.bmm(hg.transpose(1, 2), x)   # (B, H_e, C)
        He = self.edge_proj(He)                 # (B, H_e, C)

        # ---------- 6) hyper-edge -> node ----------
        # hg: (B, N, H_e), He: (B, H_e, C)
        X_new = torch.bmm(hg, He)               # (B, N, C)
        X_new = self.node_proj(X_new)           # (B, N, C)

        # ---------- 7) 通道 gate + 残差 ----------
        gated = X_new * self.gate.view(1, 1, -1)  # (B, N, C)
        x_out = x + gated                         # (B, N, C)

        # 还原回 (B, C, H, W)
        x_out_map = x_out.transpose(1, 2).view(B, C, H, W)
        return x_out_map


class ConcatenatedHyperModule(nn.Module):
    def __init__(self, in_channels, d_k=32, hyper_edge_num=16):
        super().__init__()
        self.in_channels = in_channels

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.upsample = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='bilinear')

        self.conv0 = nn.Conv2d(
            in_channels= self.in_channels[-2] + self.in_channels[-1],  # 输入通道数
            out_channels=self.in_channels[-1], # 输出通道数
            kernel_size=1,    # 1×1卷积核
            stride=1,         # 步幅=1
            padding=0,        # 无填充
            bias=True,          # 默认启用偏置
        )

        self.conv2 = nn.Conv2d(
            in_channels= self.in_channels[-2] + self.in_channels[-3],  # 输入通道数
            out_channels=self.in_channels[-3], # 输出通道数
            kernel_size=1,    # 1×1卷积核
            stride=1,         # 步幅=1
            padding=0,        # 无填充
            bias=True,          # 默认启用偏置
        )

        self.conv11 = nn.Conv2d(
            in_channels= self.in_channels[-2] * 2,  # 输入通道数
            out_channels=self.in_channels[-2], # 输出通道数
            kernel_size=1,    # 1×1卷积核
            stride=1,         # 步幅=1
            padding=0,        # 无填充
            bias=True,          # 默认启用偏置
        )

        self.d_k = d_k
        self.hyper_edge_num = hyper_edge_num

        self.conv_in = ConvNormLayer(sum(in_channels), self.in_channels[-2], kernel_size=1, stride=1, padding=0)
        self.ln = nn.LayerNorm(self.in_channels[-2])
        self.soft_hgnn = SoftHGNN2(self.in_channels[-2], d_k = d_k, hyper_edge_num=hyper_edge_num)


    def forward(self, input):
        x2, x1, x0 = input
        outputs=[]

        x2_down = self.pool2(x2)
        x0_up = self.upsample(x0)

        x = torch.cat((x0_up, x1, x2_down), dim=1) # [24, 960, 24, 40]
        # print(x.shape)
        x = self.conv_in(x)
        #hyper
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        x = self.ln(x)
        x = self.soft_hgnn(x)
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)

        # aggregate
        # x2
        out2 = self.upsample(x)
        out2 = torch.cat((x2, out2), dim=1)
        out2 = self.conv2(out2) # [24, 128, 48, 80]
        outputs.append(out2)

        # x1
        out1 = torch.cat((x1, x), dim=1)
        out1 = self.conv11(out1) # [24, 256, 24, 40]
        outputs.append(out1)

        # x0
        out0 = self.pool2(x)
        out0 = torch.cat((x0, out0), dim=1)
        out0 = self.conv0(out0) # [24, 512, 12, 20]
        outputs.append(out0)

        return outputs

if __name__ == '__main__':
    x = torch.rand([2, 256, 20, 20]).cuda()
    layer = HyperFormer(256).cuda()
    o = layer(x)


    