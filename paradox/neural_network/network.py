import collections
import numpy
from paradox.kernel.operator import Operator
from paradox.kernel.symbol import SymbolCategory, Symbol, Constant, Placeholder
from paradox.kernel.optimizer import *
from paradox.neural_network.loss import LossLayer, Loss
from paradox.neural_network.regularization import RegularizationLayer, Regularization
from paradox.neural_network.connection import ConnectionLayer, Connection
from paradox.neural_network.activation import ActivationLayer, Activation
from paradox.neural_network.convolutional_neural_network.layer import Convolution, Pooling, Unpooling
from paradox.neural_network.plugin import Plugin, default_plugin


optimizer_map = {
    'gradient descent': GradientDescentOptimizer,
    'momentum': MomentumOptimizer,
    'adaptive gradient': AdaptiveGradientOptimizer,
    'adaptive delta': AdaptiveDeltaOptimizer,
    'root mean square prop': RootMeanSquarePropOptimizer,
    'adaptive moment estimation': AdaptiveMomentEstimationOptimizer,
}


def register_optimizer(name: str, optimizer: Optimizer):
    optimizer_map[name.lower()] = optimizer


class Network:
    def __init__(self):
        self.epoch = None
        self.iteration = None
        self.epochs = None
        self.batch_size = None
        self.engine = Engine()
        self.predict_engine = Engine()
        self.__layer = []
        self.__layer_name_map = collections.OrderedDict()
        self.__layer_stack = []
        self.__input_symbol = Placeholder(name='InputSymbol')
        self.__current_symbol = self.__input_symbol
        self.__current_output = None
        self.__variables = []
        self.__layer_weights = {}
        self.__data = None
        self.__optimizer = None
        self.__loss = None
        self.__regularization_term = None
        self.__plugin = collections.OrderedDict()
        self.load_default_plugin()

    def __valid_current_output(self):
        if self.__current_output is None:
            raise ValueError('Current output is None.')
        else:
            return self.__current_output

    def get_layer(self, name: str):
        if name in self.__layer_name_map:
            return self.__layer_name_map[name]
        else:
            raise ValueError('No such layer in Network named: {}'.format(name))

    def layer_name_map(self):
        return self.__layer_name_map

    def layer_stack(self):
        return self.__layer_stack

    def add(self, layer, name=None):
        if isinstance(layer, collections.Iterable):
            for i, l in enumerate(layer):
                if name is not None and i < len(name):
                    self.__add(l, name[i])
                else:
                    self.__add(l)
        else:
            self.__add(layer, name)

    def __add(self, layer, name: str=None):
        self.__layer.append(layer)
        if name is None:
            name = 'layer_{}'.format(len(self.__layer_stack))
        if name in self.__layer_name_map:
            raise ValueError('Layer name has contained in Network: {}'.format(name))
        else:
            self.__layer_name_map[name] = layer
        self.__layer_stack.append(layer)
        if isinstance(layer, Operator):
            self.__add_operator(layer, name)
        elif isinstance(layer, ConnectionLayer):
            self.__add_connection(layer, name)
        elif isinstance(layer, Connection):
            self.__add_connection(layer.connection_layer(), name)
        elif isinstance(layer, Convolution):
            self.__add_connection(layer.convolution_layer(), name)
        elif isinstance(layer, Pooling):
            self.__add_connection(layer.pooling_layer(), name)
        elif isinstance(layer, Unpooling):
            self.__add_connection(layer.unpooling_layer(), name)
        elif isinstance(layer, ActivationLayer):
            self.__add_activation(layer, name)
        elif isinstance(layer, Activation):
            self.__add_activation(layer.activation_layer(), name)
        else:
            raise ValueError('Invalid get_layer type: {}'. format(type(layer)))

    def __add_operator(self, layer: Operator, name: str=None):
        self.__current_symbol = Symbol(operator=layer, inputs=[self.__current_symbol], category=SymbolCategory.operator)
        self.__current_output = layer.shape(self.__current_output)
        self.__layer_weights[name] = []

    def __add_connection(self, layer: ConnectionLayer, name: str=None):
        if layer.input_shape is None:
            layer.input_shape = self.__valid_current_output()
        self.__current_symbol = layer.connection(self.__current_symbol)
        self.__current_output = layer.output_shape
        for v in layer.variables():
            self.__variables.append(v)
        self.__layer_weights[name] = layer.weights()

    def __add_activation(self, layer: ActivationLayer, name: str=None):
        self.__current_symbol = layer.activation_function(self.__current_symbol)
        previous_layer = self.__layer_stack[-2]
        if isinstance(previous_layer, Connection) or isinstance(previous_layer, ConnectionLayer):
            previous_layer = previous_layer.connection_layer()
            for weight in previous_layer.weights():
                weight.value = layer.weight_initialization(weight.value.shape)
            for bias in previous_layer.biases():
                bias.value = layer.bias_initialization(bias.value.shape)
        self.__layer_weights[name] = []

    def get_symbol(self):
        return self.__current_symbol

    def optimizer(self, optimizer_object, *args, **kwargs):
        if isinstance(optimizer_object, str):
            name = optimizer_object.lower()
            if name in optimizer_map:
                self.__optimizer = optimizer_map[name](*args, **kwargs)
            else:
                raise ValueError('No such optimizer: {}'.format(name))
        elif isinstance(optimizer_object, Optimizer):
            self.__optimizer = optimizer_object
        else:
            raise ValueError('Invalid optimizer type: {}'.format(type(optimizer_object)))

    def loss(self, loss_object, *args, **kwargs):
        if isinstance(loss_object, str):
            self.__loss = Loss(loss_object, *args, **kwargs).loss_layer()
        elif isinstance(loss_object, LossLayer):
            self.__loss = loss_object
        elif isinstance(loss_object, Loss):
            self.__loss = loss_object.loss_layer()
        else:
            raise ValueError('Invalid loss type: {}'.format(type(loss_object)))

    def regularization(self, regularization_object, decay: float, name=None, *args, **kwargs):
        regularization_weights = set()
        if name is None:
            for _, weights in self.__layer_weights.items():
                regularization_weights |= set(weights)
        else:
            if isinstance(name, str):
                name = [name]
            if isinstance(name, collections.Iterable):
                for each_name in name:
                    if each_name in self.__layer_weights:
                        regularization_weights |= set(self.__layer_weights[each_name])
                    else:
                        raise ValueError('No such layer in Network named: {}'.format(each_name))
            else:
                ValueError('Invalid name type: {}'.format(type(name)))
        if isinstance(regularization_object, str):
            regularization_function = Regularization(regularization_object, *args, **kwargs).regularization_layer().regularization_term
        elif isinstance(regularization_object, RegularizationLayer):
            regularization_function = regularization_object.regularization_term
        elif isinstance(regularization_object, Regularization):
            regularization_function = regularization_object.regularization_layer().regularization_term
        else:
            raise ValueError('Invalid regularization type: {}'.format(type(regularization_object)))
        for weight in regularization_weights:
            self.__add_regularization_term(regularization_function(weight, decay))

    def __add_regularization_term(self, regularization_term):
        if self.__regularization_term is None:
            self.__regularization_term = regularization_term
        else:
            self.__regularization_term += regularization_term

    def train(self, data, target, epochs: int=10000, batch_size: int=0):
        data = numpy.array(data)
        target = numpy.array(target)
        self.epochs = epochs
        if data.shape[0] != target.shape[0]:
            raise ValueError('Data dimension not match target dimension: {} {}'.format(data.shape[0], target.shape[0]))
        data_scale = data.shape[0]
        target_symbol = None
        if batch_size != 0:
            target_symbol = Placeholder()
            loss = self.__loss.loss_function(self.__current_symbol, target_symbol)
        else:
            loss = self.__loss.loss_function(self.__current_symbol, Constant(target))
            self.engine.bind = {self.__input_symbol: data}
        if self.__regularization_term is not None:
            loss += self.__regularization_term
        self.engine.symbol = loss
        self.engine.variables = self.__variables
        try:
            self.iteration = 0
            self.run_plugin('begin_training')
            for epoch in range(self.epochs):
                self.epoch = epoch + 1
                self.run_plugin('begin_epoch')
                for i in ([0] if batch_size == 0 else range(0, data_scale, batch_size)):
                    if batch_size != 0:
                        self.engine.bind = {self.__input_symbol: data[i: min([i + batch_size, data_scale])],
                                            target_symbol: target[i: min([i + batch_size, data_scale])]}
                    self.run_plugin('begin_iteration')
                    self.__optimizer.minimize(self.engine)
                    self.iteration += 1
                    self.run_plugin('end_iteration')
                self.run_plugin('end_epoch')
        except KeyboardInterrupt:
            print('Keyboard Interrupt')
        self.run_plugin('end_training')

    def predict(self, data):
        self.predict_engine.symbol = self.__current_symbol
        self.predict_engine.bind = {self.__input_symbol: data}
        self.run_plugin('begin_predict')
        predict_data = self.predict_engine.value()
        self.run_plugin('end_predict')
        return predict_data

    def load_default_plugin(self):
        for name, plugin, enable in default_plugin:
            plugin.enable = enable
            self.add_plugin(name, plugin)

    def add_plugin(self, name: str, plugin: Plugin):
        self.__plugin[name.lower()] = plugin
        plugin.bind_network(self)

    def run_plugin(self, stage: str):
        for _, plugin in self.__plugin.items():
            if plugin.enable:
                getattr(plugin, stage)()

    def plugin(self, name: str):
        if name.lower() in self.__plugin:
            return self.__plugin[name.lower()]
        else:
            raise ValueError('No such plugin: {}'.format(name))
