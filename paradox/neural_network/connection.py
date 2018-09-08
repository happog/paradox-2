from abc import abstractmethod
import numpy
from paradox.kernel.symbol import Symbol, Variable, spread, reduce_mean
from paradox.utils.initialization import xavier_initialization, bias_initialization


class ConnectionLayer:
    def __init__(self, output_shape=None, input_shape=None):
        self._output_shape = output_shape
        self._input_shape = input_shape

    def get_output_shape(self):
        return self._output_shape

    def set_output_shape(self, output_shape):
        self._output_shape = output_shape

    output_shape = property(get_output_shape, set_output_shape)

    def get_input_shape(self):
        return self._input_shape

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    input_shape = property(get_input_shape, set_input_shape)

    def connection_layer(self):
        return self

    @abstractmethod
    def connection(self, input_symbol: Symbol):
        pass

    @abstractmethod
    def variables(self):
        pass

    def weights(self):
        return []

    def biases(self):
        return []


class Dense(ConnectionLayer):
    def __init__(self, output_dimension: int, input_dimension: int=None):
        ConnectionLayer.__init__(self, output_dimension, input_dimension)
        self.__weight = None
        self.__bias = None
        self.__output_symbol = None

    def connection(self, input_symbol: Symbol):
        if self.__output_symbol is None:
            if not isinstance(self._input_shape, int):
                input_symbol = spread(input_symbol, 1)
                self._input_shape = input_symbol.operator.shape((None,) + self._input_shape)[0][1]
            weight, bias = self.variables()
            self.__output_symbol = input_symbol @ weight + bias
        return self.__output_symbol

    def variables(self):
        if self.__weight is None:
            self.__weight = Variable(xavier_initialization((self._input_shape, self._output_shape)))
        if self.__bias is None:
            self.__bias = Variable(bias_initialization((1, self._output_shape)))
        return [self.__weight, self.__bias]

    def weights(self):
        return [self.__weight]

    def biases(self):
        return [self.__bias]


class BatchNormalization(ConnectionLayer):
    def __init__(self, input_shape=None):
        ConnectionLayer.__init__(self, input_shape, input_shape)
        self.__scale = None
        self.__shift = None
        self.__output_symbol = None
        self.__input_mean = None
        self.__input_variance = None

    def connection(self, input_symbol: Symbol):
        if self.__output_symbol is None:
            self.__input_mean = reduce_mean(input_symbol, axis=0)
            self.__input_variance = reduce_mean((input_symbol - self.__input_mean) ** 2, axis=0)
            input_normalize = (input_symbol - self.__input_mean) / (self.__input_variance + 1e-8) ** 0.5
            scale, shift = self.variables()
            self.__output_symbol = scale * input_normalize + shift
            self._output_shape = self._input_shape
        return self.__output_symbol

    def variables(self):
        if self.__scale is None:
            self.__scale = Variable(numpy.ones(self._input_shape))
        if self.__shift is None:
            self.__shift = Variable(numpy.zeros(self._input_shape))
        return [self.__scale, self.__shift]

    def normalization_symbol(self):
        return [self.__input_mean, self.__input_variance]


connection_map = {
    'dense': Dense,
    'batch normalization': BatchNormalization,
}


def register_connection(name: str, connection: ConnectionLayer):
    connection_map[name.lower()] = connection


class Connection:
    def __init__(self, name: str, *args, **kwargs):
        self.__name = name.lower()
        self.__connection = None
        if self.__name in connection_map:
            self.__connection = connection_map[self.__name](*args, **kwargs)
        else:
            raise ValueError('No such connection: {}'.format(name))

    def connection_layer(self):
        return self.__connection
