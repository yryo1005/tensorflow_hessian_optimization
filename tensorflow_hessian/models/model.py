import tensorflow as tf
from tensorflow_hessian.layers.layer import Layer

class Model:
    def __init__(self):
        self.layers = list()

        self.built = False
        self.input_shape = None
        self.output_shape = None

    def __setattr__(self, name, value):
        if isinstance(value, Layer):
            self.layers.append(value)
        if isinstance(value, tf.keras.layers.Layer):
            self.layers.append(value)
        super().__setattr__(name, value)

    def build(self, input_shape):
        self.num_params = list()
        
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Layer):
                layer.build(input_shape)
                input_shape = layer(tf.zeros(input_shape, dtype=tf.float32)).shape
                self.num_params.append(0)
            else:
                tmp_num_params, input_shape = layer.build(input_shape)
                self.num_params.append(tmp_num_params)
        self.input_shape = input_shape

        # print(tf.random.normal((sum(self.num_params), ), dtype=tf.float32))
        self.trainable_variables = [tf.Variable(tf.random.normal((sum(self.num_params), ), dtype=tf.float32),    
                                                dtype=tf.float32, trainable=True)]

        self._assign_parameters()
        self.built = True
    
    def _assign_parameters(self):
        last_index = 0
        for layer, num_param in zip(self.layers, self.num_params):
            if isinstance(layer, Layer):
                layer.assign_parameters(self.trainable_variables[0][last_index:last_index + num_param])
                last_index += num_param

    def call(self, x, training=False):
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Layer):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x
    
    def __call__(self, x, training=False):
        if not self.built:
            self.build(x.shape)
        
        if training:
            self._assign_parameters()

        return self.call(x, training=training)