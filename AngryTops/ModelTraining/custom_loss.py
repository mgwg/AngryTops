"""
This script contains custom loss functions for training
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *

kl_divergence = keras.losses.kullback_leibler_divergence

def kl_loss(y_true, y_pred):
    """
    A loss function related to the Kullback Leibler Divergence Loss Function.
    Calculates the KL Loss for each output output histogram. This will penalize
    the model for having its fitted histogram disagree with the true histogram.
    NOTE: Bigger BATCH sizes would likely work better for this.
    ===========================================================================
    HOW IT WORKS:
    Step 1: Output usually in matrix shape, ie (BATCH SIZE, 6, 4). Would want to
    "flatten" this to (BATCH SIZE, 24) for both y_true and y_pred
    Step 2: I would want to make histogram bins for each of the 24 output
    features. I will stick to the bins used in Plotting.histograms.py. Thus, I
    will use np.hist to make a (normalized) histogram for each output feature of
    shape (BATCH SIZE, # OF BINS).
    Step 3: Calculate the kl loss for each of these and take the (possibly
    weighted) sum for get the loss.
    """
    y_true = tf.constant(y_true, dtype=tf.dtypes.float64)
    y_pred = tf.constant(y_pred, dtype=tf.dtypes.float64)
    y_true = tf.reshape(y_true, shape=[y_true.shape[0], y_true.shape[1] * y_true.shape[2]])
    y_pred = tf.reshape(y_pred, shape=[y_pred.shape[0], y_pred.shape[1] * y_pred.shape[2]])
    loss = 0
    for i in range(y_true.shape[-1]):
        hist_true = tf.histogram_fixed_width(y_true[:,i],
        [tf.math.reduce_min(y_true).numpy(), tf.math.reduce_max(y_true).numpy()], nbins=10)
        hist_pred = tf.histogram_fixed_width(y_pred[:,i],
        [tf.math.reduce_min(y_pred).numpy(), tf.math.reduce_max(y_pred).numpy()], nbins=10)
        loss += kl_divergence(hist_true, hist_pred)
    return loss

if __name__ == "__main__":
    fh = np.load('/Users/fardinsyed/Desktop/Top_Quark_Project/AngryTops/CheckPoints/archived/keep/training_5_minmax/predictions.npz')
    pred = fh['pred']
    true = fh['true']
    pred(kl_loss(true[:32], pred[:32]))
