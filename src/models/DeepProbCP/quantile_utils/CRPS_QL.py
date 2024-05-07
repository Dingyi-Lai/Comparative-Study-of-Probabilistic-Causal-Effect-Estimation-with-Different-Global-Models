import numpy as np
from keras.utils import losses_utils
from keras.losses import LossFunctionWrapper
import tensorflow as tf
from typeguard import typechecked
from typing import List

# torch version: https://github.com/awslabs/gluonts/blob/74e9a39b5e2d72d0d354be1419fa38d2c8be2da1/src/gluonts/nursery/few_shot_prediction/src/meta/metrics/crps.py#L21
# noraml version: https://github.com/awslabs/gluonts/blob/19a4893233099c031c3eb868c56f3140d9b5cdb1/src/gluonts/nursery/few_shot_prediction/src/meta/metrics/numpy.py#L61

def mean_weighted_quantile_loss(
    y_pred: np.ndarray, y_true: np.ndarray, quantiles: list
) -> float:
    y_true_rep = y_true[:, None].repeat(len(quantiles), axis=1)
    quantiles = np.array([float(q) for q in quantiles])
    # print(quantiles_repeated.shape, y_pred.shape, y_true.shape)
    quantile_losses = 2 * np.sum(
        np.abs(
            (y_pred - y_true_rep)
            * ((y_true_rep <= y_pred) - quantiles[:, None])
        ),
        axis=-1,
    )  # shape [num_time_series, num_quantiles]
    denom = np.sum(np.abs(y_true_rep))  # shape [1]
    weighted_losses = quantile_losses.sum(0) / denom  # shape [num_quantiles]
    return weighted_losses

@tf.function
def pinball_loss(
    y_true, y_pred, tau = 0.5
):
    """Computes the pinball loss between `y_true` and `y_pred`.

    `loss = maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred))`

    In the context of regression this loss yields an estimator of the tau
    conditional quantile.

    See: https://en.wikipedia.org/wiki/Quantile_regression

    Usage:

    >>> loss = tfa.losses.pinball_loss([0., 0., 1., 1.],
    ... [1., 1., 1., 0.], tau=.1)
    >>> loss
    <tf.Tensor: shape=(), dtype=float32, numpy=0.475>

    Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
    tau: (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
        shape = `[d0,..., dn]`.  It defines the slope of the pinball loss. In
        the context of quantile regression, the value of tau determines the
        conditional quantile level. When tau = 0.5, this amounts to l1
        regression, an estimator of the conditional median (0.5 quantile).

    Returns:
        pinball_loss: 1-D float `Tensor` with shape [batch_size].

    References:
    - https://en.wikipedia.org/wiki/Quantile_regression
    - https://projecteuclid.org/download/pdfview_1/euclid.bj/1297173840
    """
    # under_bias = q  * K.maximum(y_true - y_pred_q, 0)
    # over_bias = (1 - q) * K.maximum(y_pred_q - y_true, 0)

    # qt_loss = under_bias + over_bias
    # print(np.sum(np.abs((y_pred_q - y_true) * ((y_true <= y_pred_q) - q))))
    # print(qt_loss)
    # assert qt_loss == np.sum(np.abs((y_pred_q - y_true) * ((y_true <= y_pred_q) - q)))
    # return qt_loss
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Broadcast the pinball slope along the batch dimension
    tau = tf.expand_dims(tf.cast(tau, y_pred.dtype), 0)
    one = tf.cast(1, tau.dtype)

    delta_y = y_true - y_pred
    pinball = tf.math.maximum(tau * delta_y, (tau - one) * delta_y)
    return tf.reduce_mean(pinball, axis=-1)


class PinballLoss(LossFunctionWrapper):
    """Computes the pinball loss between `y_true` and `y_pred`.

    `loss = maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred))`

    In the context of regression, this loss yields an estimator of the tau
    conditional quantile.

    See: https://en.wikipedia.org/wiki/Quantile_regression

    Usage:

    >>> pinball = tfa.losses.PinballLoss(tau=.1)
    >>> loss = pinball([0., 0., 1., 1.], [1., 1., 1., 0.])
    >>> loss
    <tf.Tensor: shape=(), dtype=float32, numpy=0.475>

    Usage with the `tf.keras` API:

    >>> model = tf.keras.Model()
    >>> model.compile('sgd', loss=tfa.losses.PinballLoss(tau=.1))

    Args:
      tau: (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
        shape = `[d0,..., dn]`.  It defines the slope of the pinball loss. In
        the context of quantile regression, the value of tau determines the
        conditional quantile level. When tau = 0.5, this amounts to l1
        regression, an estimator of the conditional median (0.5 quantile).
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`.
        When used with `tf.distribute.Strategy`, outside of built-in training
        loops such as `tf.keras` `compile` and `fit`, using `AUTO` or
        `SUM_OVER_BATCH_SIZE` will raise an error. Please see
        https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
        for more details on this.
      name: Optional name for the op.

    References:
      - https://en.wikipedia.org/wiki/Quantile_regression
      - https://projecteuclid.org/download/pdfview_1/euclid.bj/1297173840
    """

    @typechecked
    def __init__(
        self,
        tau = 0.5,
        reduction=losses_utils.ReductionV2.AUTO,
        name: str = "pinball_loss",
    ):
        super().__init__(pinball_loss, reduction=reduction, name=name, tau=tau)
