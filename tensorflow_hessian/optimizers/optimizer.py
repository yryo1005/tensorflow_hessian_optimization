class Optimizer:
    def __init__(self):
        pass

    def build(self, var):
        """
            var: [tf.Variable], shape=(num_params,)
        """
        pass

    def apply_gradients(self, vars, grads, hessians = None):
        """
            vars: [tf.Variable], shape=(num_params,)
            grads: [tf.Tensor], shape=(num_params,)
            hessians: [tf.Tensor], shape=(num_params, num_params) or None
        """

        for i in range(len(vars)):
            self.update_step(vars[i], grads[i], hessians[i] if hessians is not None else None)

    def update_step(self, var, grad, hessian = None):
        """
            var: tf.Variable, shape=(num_params,)
            grad: tf.Tensor, shape=(num_params,)
            hessian: tf.Tensor, shape=(num_params, num_params) or None
        """
        pass