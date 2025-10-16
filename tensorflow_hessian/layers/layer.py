import tensorflow as tf

class Layer:
    # tfh.Layerの基底クラス
    def __init__(self):
        self.built = False
        self.input_shape = None
        self.output_shape = None
    
    def build(self, input_shape):
        pass

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

    def __call__(self, x):
        return self.forward(x)