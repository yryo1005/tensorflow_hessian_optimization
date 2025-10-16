import tensorflow as tf
from tensorflow_hessian.layers.layer import Layer

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="valid"):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.upper()
        self.W = None
        self.b = None
    
    def build(self, input_shape):
        """
            input_shape: (bs, height, width, channels)
        """
        num_params = 0
        num_params += self.kernel_size[0] * self.kernel_size[1] * input_shape[-1] * self.filters  # weights
        num_params += self.filters  # biases

        self.input_shape = input_shape
        out_height = (input_shape[1] - self.kernel_size[0]) // self.strides[0] + 1 if self.padding == "VALID" else input_shape[1] // self.strides[0]
        out_width = (input_shape[2] - self.kernel_size[1]) // self.strides[1] + 1 if self.padding == "VALID" else input_shape[2] // self.strides[1]
        self.output_shape = (input_shape[0], out_height, out_width, self.filters)

        self.built = True
        return num_params, self.output_shape

    def assign_parameters(self, params):
        input_channels = self.input_shape[-1]
        weights_shape = (self.kernel_size[0], self.kernel_size[1], input_channels, self.filters)
        biases_shape = (self.filters,)

        weights_size = self.kernel_size[0] * self.kernel_size[1] * input_channels * self.filters
        biases_size = self.filters

        self.W = tf.reshape(params[:weights_size], weights_shape)
        self.b = tf.reshape(params[weights_size:weights_size + biases_size], biases_shape)
    
    def forward(self, x):
        if not self.built:
            raise RuntimeError("Layer is not built yet.")
        x = tf.nn.conv2d(x, self.W, strides=[1, self.strides[0], self.strides[1], 1], padding=self.padding)
        return x + self.b