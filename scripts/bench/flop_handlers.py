from collections import Counter
import math
from functools import reduce
from operator import mul


def _prod(values):
    if values is None:
        return None
    result = 1
    for value in values:
        if value is None:
            return None
        result *= int(value)
    return result


def _shape(value):
    from fvcore.nn.jit_handles import get_shape

    result = get_shape(value)
    if result is None:
        return None
    return [int(x) if x is not None else None for x in result]


def _numel(value):
    return _prod(_shape(value))


def _to_python(value, default=None):
    try:
        result = value.toIValue()
    except Exception:
        result = None

    if result is not None:
        return result

    try:
        node = value.node()
    except Exception:
        return default

    if node.kind() != "prim::ListConstruct":
        return default

    items = []
    for item in node.inputs():
        try:
            items.append(item.toIValue())
        except Exception:
            return default
    return items


def _as_tuple(value, ndim=2, default=1):
    value = _to_python(value, None)
    if value is None:
        return tuple([default] * ndim)
    if isinstance(value, int):
        return tuple([value] * ndim)
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    return tuple([default] * ndim)


def _counter(name, value):
    if value is None:
        value = 0
    return Counter({name: int(max(0, round(value)))})


def _output_elementwise(name, scale=1):
    def handle(_inputs, outputs):
        return _counter(name, _numel(outputs[0]) * scale)

    return handle


def _zero_flops(name):
    def handle(_inputs, _outputs):
        return _counter(name, 0)

    return handle


def _reduction_flops(name, include_div=False):
    def handle(inputs, outputs):
        input_numel = _numel(inputs[0])
        output_numel = _numel(outputs[0])
        if input_numel is None or output_numel is None:
            return _counter(name, 0)

        flops = max(input_numel - output_numel, 0)
        if include_div:
            flops += output_numel
        return _counter(name, flops)

    return handle


def _softmax_flops(inputs, outputs):
    del inputs
    output_numel = _numel(outputs[0])
    # exp + reduction add + divide, approximate.
    return _counter("softmax", 3 * output_numel)


def _pool_flops(name, reduction_ops_per_window):
    def handle(inputs, outputs):
        output_numel = _numel(outputs[0])
        kernel = _as_tuple(inputs[1], ndim=2, default=1)
        kernel_ops = reduce(mul, kernel, 1)
        if reduction_ops_per_window:
            kernel_ops = max(kernel_ops - 1, 0)
        return _counter(name, output_numel * kernel_ops)

    return handle


def _convolution_mode_flops(inputs, outputs):
    input_shape = _shape(inputs[0])
    weight_shape = _shape(inputs[1])
    output_shape = _shape(outputs[0])
    if input_shape is None or weight_shape is None or output_shape is None:
        return _counter("conv", 0)

    batch_size = output_shape[0]
    output_channels = output_shape[1]
    output_spatial = _prod(output_shape[2:])
    kernel_mul = _prod(weight_shape[1:])
    flops = batch_size * output_channels * output_spatial * kernel_mul
    return _counter("conv", flops)


def _topk_flops(inputs, outputs):
    input_shape = _shape(inputs[0])
    output_shape = _shape(outputs[0])
    if input_shape is None or output_shape is None:
        return _counter("topk", 0)

    dim = _to_python(inputs[2], -1) if len(inputs) > 2 else -1
    if dim is None:
        dim = -1
    dim = int(dim)
    if dim < 0:
        dim += len(input_shape)

    n = input_shape[dim]
    k = output_shape[dim]
    input_numel = _prod(input_shape)
    if n in (None, 0) or k in (None, 0) or input_numel is None:
        return _counter("topk", 0)

    slices = input_numel / n
    flops = slices * n * math.log2(max(k, 2))
    return _counter("topk", flops)


def _rfft_flops(inputs, outputs):
    del outputs
    input_shape = _shape(inputs[0])
    if input_shape is None:
        return _counter("fft_rfft", 0)

    dim = _to_python(inputs[2], -1) if len(inputs) > 2 else -1
    if dim is None:
        dim = -1
    dim = int(dim)
    if dim < 0:
        dim += len(input_shape)

    n = _to_python(inputs[1], None) if len(inputs) > 1 else None
    if n is None:
        n = input_shape[dim]
    n = int(n)
    input_numel = _prod(input_shape)
    if n <= 1 or input_numel is None:
        return _counter("fft_rfft", 0)

    transforms = input_numel / n
    # Common real FFT estimate. Exact cost depends on the backend/kernel.
    flops = 2.5 * transforms * n * math.log2(n)
    return _counter("fft_rfft", flops)


def add_complete_flop_handles(flops):
    """Add handlers for ops that fvcore commonly leaves unsupported.

    The counts are estimates under fvcore's convention that one fused
    multiply-add is one flop. Shape-only ops are registered as zero flops so
    they do not remain in the unsupported list.
    """

    handles = {
        "aten::_convolution_mode": _convolution_mode_flops,
        "aten::max_pool2d": _pool_flops("max_pool2d", True),
        "aten::avg_pool2d": _pool_flops("avg_pool2d", False),
        "aten::add": _output_elementwise("add"),
        "aten::add_": _output_elementwise("add"),
        "aten::sub": _output_elementwise("sub"),
        "aten::rsub": _output_elementwise("sub"),
        "aten::mul": _output_elementwise("mul"),
        "aten::div": _output_elementwise("div"),
        "aten::abs": _output_elementwise("abs"),
        "aten::sigmoid": _output_elementwise("sigmoid"),
        "aten::tanh": _output_elementwise("tanh"),
        "aten::gelu": _output_elementwise("gelu"),
        "aten::silu_": _output_elementwise("silu"),
        "aten::log": _output_elementwise("log"),
        "aten::softmax": _softmax_flops,
        "aten::sum": _reduction_flops("sum"),
        "aten::mean": _reduction_flops("mean", include_div=True),
        "aten::fft_rfft": _rfft_flops,
        "aten::topk": _topk_flops,
        "aten::repeat": _zero_flops("repeat"),
        "aten::unflatten": _zero_flops("unflatten"),
    }

    for name, handle in handles.items():
        flops.set_op_handle(name, handle)
    return flops
