from abc import abstractmethod
from functools import reduce
from paradox.kernel.symbol import *


def element_wise_shape(*shape_list):
    broadcast_map = {_shape: [] for _shape in shape_list}
    new_shape = []
    for shape in shape_list:
        if len(shape) > len(new_shape):
            new_shape = list(shape)
    for i in range(-len(new_shape), 0):
        index = len(new_shape) + i
        dimensions = {}
        for shape in shape_list:
            if -i > len(shape):
                broadcast_map[shape].append(-1)
            else:
                broadcast_map[shape].append(0)
                dimensions[shape] = shape[i]
        new_shape[index] = max([_d for _, _d in dimensions.items()])
        for shape, dimension in dimensions.items():
            if dimension != new_shape[index]:
                if dimension == 1:
                    broadcast_map[shape][-1] = 1
                else:
                    raise ValueError('Can not broadcast these shapes: {}'.format(shape_list))
    return (tuple(new_shape),) + tuple(tuple(broadcast_map[_shape]) for _shape in shape_list)


def matrix_multiply_shape(shape_a, shape_b):
    try:
        if len(shape_a) == 0 or len(shape_b) == 0:
            raise ValueError()
        if len(shape_a) == 1 and len(shape_b) == 1:
            if shape_a[0] == shape_b[0]:
                return (), (), ()
            else:
                raise ValueError()
        if len(shape_a) == 1:
            if shape_a[0] == shape_b[-2]:
                new_shape = list(shape_b)
                del new_shape[-2]
                return tuple(new_shape), (), ()
            else:
                raise ValueError()
        if len(shape_b) == 1:
            if shape_a[-1] == shape_b[0]:
                return shape_a[:-1], (), ()
            else:
                raise ValueError()
        if shape_a[-1] == shape_b[-2]:
            gap = abs(len(shape_a) - len(shape_b))
            if len(shape_a) > len(shape_b):
                if shape_a[gap:-2] != shape_b[:-2]:
                    raise ValueError()
                new_shape = list(shape_a)
                broadcast_a = (0,) * len(shape_a)
                broadcast_b = shape_a[:gap] + (0, 0)
            else:
                if shape_b[gap:-2] != shape_a[:-2]:
                    raise ValueError()
                new_shape = list(shape_b)
                broadcast_a = shape_b[:gap] + (0, 0)
                broadcast_b = (0,) * len(shape_b)
            new_shape[-1] = shape_b[-1]
            new_shape[-2] = shape_a[-2]
            return tuple(new_shape), broadcast_a, broadcast_b
        else:
            raise ValueError()
    except ValueError:
        raise ValueError('Can not execute matrix multiply with these two shapes: a={}, b={}'.format(shape_a, shape_b))


def reduce_shape(shape_a, axis, invariant):
    if axis is None:
        return (), ()
    else:
        new_shape = list(shape_a)
        if invariant:
            new_shape[axis] = 1
        else:
            del new_shape[axis]
        return tuple(new_shape), ()


def transpose_shape(shape_a, axes):
    if axes is None:
        return tuple(reversed(shape_a)), ()
    else:
        if len(set(axes)) == len(axes):
            if set(axes) == set(range(len(axes))) and len(axes) == len(shape_a):
                new_shape = [0] * len(shape_a)
                for i, d in zip(axes, shape_a):
                    new_shape[i] = d
                return tuple(new_shape), ()
            else:
                raise ValueError('Invalid axes for this Shape: shape={}, axes={}'.format(shape_a, axes))
        else:
            raise ValueError('Repeated axis in axes: {}'.format(axes))


def concatenate_shape(axis, *shape_list):
    new_shape = list(shape_list[0])
    shape_len = len(new_shape)
    for shape in shape_list[1:]:
        if len(shape) == len(shape):
            for i in range(shape_len):
                if i == axis:
                    new_shape[i] += shape[i]
                else:
                    if new_shape[i] != shape[i]:
                        raise ValueError('Concatenate shape not match: {}'.format(shape_list))
        else:
            raise ValueError('All shapes must have same dimensions')
    return (tuple(new_shape),) + () * len(shape_list)


def slice_shape(shape_a, slice_list):
    if not isinstance(slice_list, list):
        slice_list = [slice_list]
    new_shape = list(shape_a)
    delete_dimension = 0
    for i, s in enumerate(slice_list):
        index = i - delete_dimension
        if index < len(shape_a):
            if isinstance(s, slice):
                new_shape[index] = len(([0] * shape_a[index])[s])
            elif isinstance(s, int):
                del new_shape[index]
                delete_dimension += 1
            else:
                raise ValueError('Invalid slice type: {}'.format(type(s)))
        else:
            raise ValueError('Shape not match slice: shape={} slice={}'.format(shape_a, slice_list))
    return tuple(new_shape), ()


def rotate90_shape(shape_a, count, axes):
    new_shape = list(shape_a)
    if count % 2 == 0:
        return tuple(new_shape), ()
    else:
        new_shape[axes[0]], new_shape[axes[1]] = new_shape[axes[1]], new_shape[axes[0]]
        return tuple(new_shape), ()


class Operator:
    operator_sign = None
    inputs_count = None
    auto_reduce = True
    arguments = {}

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join(['{}={}'.format(key, value) for key, value in self.arguments.items()]))

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    @abstractmethod
    def gradient(self, *args, **kwargs):
        pass

    @abstractmethod
    def shape(self, *args, **kwargs):
        pass


class Negative(Operator):
    def __init__(self):
        self.operator_sign = '-'
        self.inputs_count = 1

    def compute(self, value_a):
        return -value_a

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: -forward]

    def shape(self, shape_a):
        return shape_a, ()


class Absolute(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.absolute(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * where(symbol_a > 0, 1, -1)]

    def shape(self, shape_a):
        return shape_a, ()


class Plus(Operator):
    def __init__(self):
        self.operator_sign = '+'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return value_a + value_b

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * Constant(1),
                lambda: forward * Constant(1)]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Subtract(Operator):
    def __init__(self):
        self.operator_sign = '-'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return value_a - value_b

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * Constant(1),
                lambda: forward * Constant(-1)]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Multiply(Operator):
    def __init__(self):
        self.operator_sign = '*'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return value_a * value_b

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * symbol_b,
                lambda: forward * symbol_a]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Divide(Operator):
    def __init__(self):
        self.operator_sign = '/'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return value_a / value_b

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * Constant(1) / symbol_b,
                lambda: forward * Constant(-1) * symbol_a / (symbol_b ** Constant(2))]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class MatrixMultiply(Operator):
    def __init__(self):
        self.operator_sign = '@'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        if len(value_a.shape) == 0 or len(value_b.shape) == 0:
            return value_a * value_b
        else:
            return value_a @ value_b

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        shape_a = engine.shape(symbol_a)
        shape_b = engine.shape(symbol_b)
        if len(shape_a) >= 2:
            axes_a = tuple(range(len(shape_a) - 2)) + (-1, -2)
        else:
            axes_a = None
        if len(shape_b) >= 2:
            axes_b = tuple(range(len(shape_b) - 2)) + (-1, -2)
        else:
            axes_b = None
        return [lambda: forward @ transpose(symbol_b, axes=axes_a),
                lambda: transpose(symbol_a, axes=axes_b) @ forward]

    def shape(self, shape_a, shape_b):
        return matrix_multiply_shape(shape_a, shape_b)


class Transpose(Operator):
    def __init__(self, axes=None):
        self.inputs_count = 1
        self.arguments = {'axes': axes}

    def compute(self, value_a):
        return numpy.transpose(value_a, axes=self.arguments['axes'])

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: transpose(forward, axes=self.arguments['axes'])]

    def shape(self, shape_a):
        return transpose_shape(shape_a, self.arguments['axes'])


class ReduceSum(Operator):
    def __init__(self, axis: int=None, invariant: bool=False):
        self.inputs_count = 1
        self.arguments = {'axis': axis, 'invariant': invariant}

    def compute(self, value_a):
        return numpy.sum(value_a, axis=self.arguments['axis'], keepdims=self.arguments['invariant'])

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        shape_a = engine.shape(symbol_a)
        axis = self.arguments['axis']
        if axis:
            return [lambda: broadcast(expand(forward, axis), shape_a)]
        else:
            return [lambda: broadcast(forward, shape_a)]

    def shape(self, shape_a):
        return reduce_shape(shape_a, **self.arguments)


class ReduceMean(Operator):
    def __init__(self, axis: int=None, invariant: bool=False):
        self.inputs_count = 1
        self.arguments = {'axis': axis, 'invariant': invariant}

    def compute(self, value_a):
        return numpy.mean(value_a, axis=self.arguments['axis'], keepdims=self.arguments['invariant'])

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        shape_a = engine.shape(symbol_a)
        axis = self.arguments['axis']
        if axis:
            return [lambda: broadcast(expand(forward, axis), shape_a) / shape_a[axis]]
        else:
            return [lambda: broadcast(forward, shape_a) / reduce(lambda x, y: x * y, shape_a, 1)]

    def shape(self, shape_a):
        return reduce_shape(shape_a, **self.arguments)


class Expand(Operator):
    def __init__(self, axis: int):
        self.inputs_count = 1
        self.arguments = {'axis': axis}

    def compute(self, value_a):
        return numpy.expand_dims(value_a, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: reduce_mean(forward, **self.arguments)]

    def shape(self, shape_a):
        new_shape = list(shape_a)
        new_shape.insert(self.arguments['axis'], 1)
        broadcast_a = [0] * len(shape_a)
        broadcast_a.insert(self.arguments['axis'], 1)
        return tuple(new_shape), tuple(broadcast_a)


class Broadcast(Operator):
    def __init__(self, shape):
        self.inputs_count = 1
        self.arguments = {'shape': shape}

    def compute(self, value_a):
        if len(value_a.shape) == 1 and len(self.arguments['shape']) > 1:
            if value_a.shape[0] == self.arguments['shape'][0]:
                value_a = value_a.reshape((value_a.shape[0], 1))
        return numpy.broadcast_to(value_a, **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward]

    def shape(self, shape_a):
        return element_wise_shape(shape_a, self.arguments['shape'])[:2]


class Power(Operator):
    def __init__(self):
        self.operator_sign = '**'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return numpy.power(value_a, value_b)

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * symbol_b * (symbol_a ** (symbol_b - Constant(1))),
                lambda: forward * (symbol_a ** symbol_b) * log(symbol_a)]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Log(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.log(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * Constant(1) / symbol_a]

    def shape(self, shape_a):
        return shape_a, ()


class Where(Operator):
    def __init__(self):
        self.inputs_count = 3

    def compute(self, value_condition, value_a, value_b):
        return numpy.array(numpy.where(value_condition, value_a, value_b), dtype=float)

    def gradient(self, engine, symbol_forward, symbol_condition, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: Constant(0),
                lambda: forward * where(symbol_condition, forward, Constant(0)),
                lambda: forward * where(symbol_condition, Constant(0), forward)]

    def shape(self, shape_condition, shape_a, shape_b):
        return element_wise_shape(shape_condition, shape_a, shape_b)


class Equal(Operator):
    def __init__(self):
        self.operator_sign = '=='
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return numpy.equal(value_a, value_b)

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: where(symbol_a == symbol_b, forward, Constant(0)),
                lambda: where(symbol_a == symbol_b, forward, Constant(0))]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class NotEqual(Operator):
    def __init__(self):
        self.operator_sign = '!='
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return numpy.not_equal(value_a, value_b)

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: where(symbol_a != symbol_b, forward, Constant(0)),
                lambda: where(symbol_a != symbol_b, forward, Constant(0))]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Less(Operator):
    def __init__(self):
        self.operator_sign = '<'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return numpy.less(value_a, value_b)

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: where(symbol_a < symbol_b, forward, Constant(0)),
                lambda: where(symbol_a < symbol_b, forward, Constant(0))]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class LessEqual(Operator):
    def __init__(self):
        self.operator_sign = '<='
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return numpy.less_equal(value_a, value_b)

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: where(symbol_a <= symbol_b, forward, Constant(0)),
                lambda: where(symbol_a <= symbol_b, forward, Constant(0))]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Greater(Operator):
    def __init__(self):
        self.operator_sign = '>'
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return numpy.greater(value_a, value_b)

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: where(symbol_a > symbol_b, forward, Constant(0)),
                lambda: where(symbol_a > symbol_b, forward, Constant(0))]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class GreaterEqual(Operator):
    def __init__(self):
        self.operator_sign = '>='
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return numpy.greater_equal(value_a, value_b)

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: where(symbol_a >= symbol_b, forward, Constant(0)),
                lambda: where(symbol_a >= symbol_b, forward, Constant(0))]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Maximum(Operator):
    def __init__(self):
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return numpy.maximum(value_a, value_b)

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: where(symbol_a > symbol_b, forward, Constant(0)),
                lambda: where(symbol_b > symbol_a, forward, Constant(0))]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Minimum(Operator):
    def __init__(self):
        self.inputs_count = 2

    def compute(self, value_a, value_b):
        return numpy.minimum(value_a, value_b)

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: where(symbol_a < symbol_b, forward, Constant(0)),
                lambda: where(symbol_b < symbol_a, forward, Constant(0))]

    def shape(self, shape_a, shape_b):
        return element_wise_shape(shape_a, shape_b)


class Sine(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.sin(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * cos(symbol_a)]

    def shape(self, shape_a):
        return shape_a, ()


class Cosine(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.cos(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * -sin(symbol_a)]

    def shape(self, shape_a):
        return shape_a, ()


class Tangent(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.tan(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * Constant(1) / cos(symbol_a) ** 2]

    def shape(self, shape_a):
        return shape_a, ()


class ArcSine(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.arcsin(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * Constant(1) / (Constant(1) - symbol_a ** Constant(2)) ** 0.5]

    def shape(self, shape_a):
        return shape_a, ()


class ArcCosine(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.arccos(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * -Constant(1) / (Constant(1) - symbol_a ** Constant(2)) ** 0.5]

    def shape(self, shape_a):
        return shape_a, ()


class ArcTangent(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.arctan(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * Constant(1) / (Constant(1) + symbol_a ** Constant(2))]

    def shape(self, shape_a):
        return shape_a, ()


class HyperbolicSine(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.sinh(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * cosh(symbol_a)]

    def shape(self, shape_a):
        return shape_a, ()


class HyperbolicCosine(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.cosh(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * sinh(symbol_a)]

    def shape(self, shape_a):
        return shape_a, ()


class HyperbolicTangent(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.tanh(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * (Constant(1) - tanh(symbol_a) ** Constant(2))]

    def shape(self, shape_a):
        return shape_a, ()


class HyperbolicArcSine(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.arcsinh(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * Constant(1) / (symbol_a ** Constant(2) + Constant(1)) ** Constant(0.5)]

    def shape(self, shape_a):
        return shape_a, ()


class HyperbolicArcCosine(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.arccosh(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * Constant(1) / (symbol_a ** Constant(2) - Constant(1)) ** Constant(0.5)]

    def shape(self, shape_a):
        return shape_a, ()


class HyperbolicArcTangent(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.arctanh(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * Constant(1) / (Constant(1) - symbol_a ** Constant(2))]

    def shape(self, shape_a):
        return shape_a, ()


class Exponential(Operator):
    def __init__(self):
        self.inputs_count = 1

    def compute(self, value_a):
        return numpy.exp(value_a)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: forward * exp(symbol_a)]

    def shape(self, shape_a):
        return shape_a, ()


class SliceAssign(Operator):
    def __init__(self, slice_tuple):
        self.inputs_count = 2
        self.arguments = {'slice_tuple': slice_tuple}

    def compute(self, value_a, value_b):
        value_a[self.arguments['slice_tuple']] = value_b
        return value_a

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        slice_tuple = self.arguments['slice_tuple']
        forward = engine.gradient(symbol_forward)
        return [lambda: slice_assign(forward, Constant(numpy.zeros(engine.shape(symbol_b))), slice_tuple),
                lambda: forward[slice_tuple]]

    def shape(self, shape_a, shape_b):
        shape_select = slice_shape(shape_a, self.arguments['slice_tuple'])[0]
        shape_package = element_wise_shape(shape_select, shape_b)
        if shape_package[0] != shape_select:
            raise ValueError('Can not assign: {} to {} with {}'.format(shape_b, shape_a, self.arguments['slice_tuple']))
        return shape_a, (), shape_package[2]


class SliceSelect(Operator):
    def __init__(self, slice_tuple):
        self.inputs_count = 1
        self.arguments = {'slice_tuple': slice_tuple}

    def compute(self, value_a):
        return value_a[self.arguments['slice_tuple']]

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        symbol_zero = Constant(numpy.zeros(engine.shape(symbol_a)))
        return [lambda: slice_assign(symbol_zero, self.arguments['slice_tuple'], forward)]

    def shape(self, shape_a):
        return slice_shape(shape_a, self.arguments['slice_tuple'])


class Concatenate(Operator):
    def __init__(self, axis: int=0):
        self.inputs_count = 2
        self.arguments = {'axis': axis}

    def compute(self, value_a, value_b):
        return numpy.concatenate((value_a, value_b), **self.arguments)

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        split_dimension = engine.shape(symbol_a)[self.arguments['axis']]
        total_dimension = engine.shape(symbol_forward)[self.arguments['axis']]
        return [lambda: forward[[slice(None)] * self.arguments['axis'] + [slice(0, split_dimension)]],
                lambda: forward[[slice(None)] * self.arguments['axis'] + [slice(split_dimension, total_dimension)]]]

    def shape(self, *shapes):
        return concatenate_shape(self.arguments['axis'], *shapes)


class Rotate90(Operator):
    def __init__(self, count: int=1, axes: tuple=None):
        self.inputs_count = 1
        self.arguments = {'count': count, 'axes': axes}

    def compute(self, value_a):
        return numpy.rot90(value_a, k=self.arguments['count'], axes=self.arguments['axes'])

    def gradient(self, engine, symbol_forward, symbol_a, symbol_b):
        forward = engine.gradient(symbol_forward)
        return [lambda: rotate90(forward, -self.arguments['count'] % 4, self.arguments['axes'])]

    def shape(self, shape_a):
        return rotate90_shape(shape_a, **self.arguments)


class Flip(Operator):
    def __init__(self, axis: int):
        self.inputs_count = 1
        self.arguments = {'axis': axis}

    def compute(self, value_a):
        return numpy.flip(value_a, self.arguments['axis'])

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: flip(forward, self.arguments['axis'])]

    def shape(self, shape_a):
        return shape_a, ()


class Reshape(Operator):
    def __init__(self, shape):
        self.inputs_count = 1
        self.arguments = {'shape': shape}

    def compute(self, value_a):
        return numpy.reshape(value_a, self.arguments['shape'])

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: reshape(forward, engine.shape(symbol_a))]

    def shape(self, shape_a):
        return self.arguments['shape'], ()


class Spread(Operator):
    def __init__(self, position):
        self.inputs_count = 1
        self.arguments = {'position': position}

    def compute(self, value_a):
        shape_a = value_a.shape
        spread_dimension = reduce(lambda a, b: a * b, shape_a[self.arguments['position']:])
        new_shape = shape_a[:self.arguments['position']] + (spread_dimension,)
        return numpy.reshape(value_a, new_shape)

    def gradient(self, engine, symbol_forward, symbol_a):
        forward = engine.gradient(symbol_forward)
        return [lambda: reshape(forward, engine.shape(symbol_a))]

    def shape(self, shape_a):
        spread_dimension = reduce(lambda a, b: a * b, shape_a[self.arguments['position']:])
        new_shape = shape_a[:self.arguments['position']] + (spread_dimension,)
        return new_shape, ()
