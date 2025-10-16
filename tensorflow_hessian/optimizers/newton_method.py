import tensorflow as tf
from tensorflow_hessian.optimizers.optimizer import Optimizer

class NewtonMethod(Optimizer):
    def __init__(self, eta =0.001, alpha = 0.1):
        """
            eta: 学習率
            alpha: ヘッセ行列の正則化パラメータ
                   alpha = 0 のとき、ヘッセ行列の正則化を行わない
        """
        self.eta = eta
        self.alpha = alpha

    def update_step(self, var, grad, hessian):
        """
            var: tf.Variable, shape=(num_params,)
            grad: tf.Tensor, shape=(num_params,)
            hessian: tf.Tensor, shape=(num_params, num_params) or None
        """

        # inv_hessian = tf.linalg.inv(hessian + self.alpha * tf.eye(hessian.shape[0]))
        # delta = self.eta * (inv_hessian @ grad[..., tf.newaxis])[:, 0]
        reg_hessian = hessian + self.alpha * tf.eye(hessian.shape[0], dtype=hessian.dtype)
        delta = tf.linalg.solve(reg_hessian, self.eta * grad[..., tf.newaxis])[:, 0]
        var.assign_sub(delta)  # var = var - delta