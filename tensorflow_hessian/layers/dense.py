import tensorflow as tf
from tensorflow_hessian.layers.layer import Layer

class Dense(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.W = None
        self.b = None
    
    def build(self, input_shape):
        """
            input_shape: (bs, input_dim)
        """
        num_params = 0
        num_params += input_shape[-1] * self.units  # weights
        num_params += self.units  # biases

        self.input_shape = input_shape
        self.output_shape = (input_shape[0], self.units)

        self.built = True
        return num_params, self.output_shape

    def assign_parameters(self, params):
        input_dim = self.input_shape[-1]
        output_dim = self.units

        weights_shape = (input_dim, output_dim)
        biases_shape = (output_dim)

        weights_size = input_dim * output_dim
        biases_size = output_dim

        self.W = tf.reshape(params[:weights_size], weights_shape)
        self.b = tf.reshape(params[weights_size:weights_size + biases_size], biases_shape)

    def forward(self, x):
        if not self.built:
            raise RuntimeError("Layer is not built yet.")
        return tf.matmul(x, self.W) + self.b