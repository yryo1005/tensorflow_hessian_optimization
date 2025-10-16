import tensorflow as tf
from tensorflow_hessian.optimizers.optimizer import Optimizer

class MomentumNewtonMethod(Optimizer):
    def __init__(self, eta =0.001, mu = 0.9, alpha = 0.1):
        """
            eta: 学習率
            mu: モーメンタム係数
            alpha: ヘッセ行列の正則化パラメータ
                   alpha = 0 のとき、ヘッセ行列の正則化を行わない
        """
        self.eta = eta
        self.mu = mu
        self.alpha = alpha
        self.velocity = None

    def update_step(self, var, grad, hessian):
        """
            var: tf.Variable, shape=(num_params,)
            grad: tf.Tensor, shape=(num_params,)
            hessian: tf.Tensor, shape=(num_params, num_params) or None
        """
        if self.velocity is None:
            self.velocity = tf.zeros_like(var)
        inv_hessian = tf.linalg.inv(hessian + self.alpha * tf.eye(hessian.shape[0]))
        delta = self.eta * (inv_hessian @ grad[..., tf.newaxis])[:, 0]
        self.velocity = self.mu * self.velocity - delta
        var.assign_add(self.velocity)