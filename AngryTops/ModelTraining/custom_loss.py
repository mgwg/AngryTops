"""
This script contains custom loss functions and metrics
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import tensorflow.keras.backend as K

def weighted_MSE1(y_true, y_pred):
    """
    @Description
    A weighted MSE Loss. This loss function places a heavier weight on px and py
    values and uses that as the weight.
    Expected input is (6 x 3). 6 particles. (px, py, pz) for each.
    ====================================================================
    @Input Format
    "target_W_had_Px", "target_W_had_Py", "target_W_had_Pz",
    "target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz",
    "target_b_had_Px", "target_b_had_Py", "target_b_had_Pz",
    "target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz",
    "target_t_had_Px", "target_t_had_Py", "target_t_had_Pz",
    "target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz"
    ====================================================================
    """
    loss = tf.constant(0.0)
    loss += tf.math.reduce_sum(tf.math.squared_difference(y_true[:2],y_pred[:2]))
    loss += 5 * tf.math.reduce_sum(tf.math.squared_difference(y_true[2:4,:2],y_pred[2:4,:2]))
    loss += tf.math.reduce_sum(tf.math.squared_difference(y_true[2:4,2],y_pred[2:4,2]))
    loss += tf.math.reduce_sum(tf.math.squared_difference(y_true[2:],y_pred[2:]))
    return loss / (1 * 14 + 5 * 4)

def weighted_MSE2(y_true, y_pred):
    """
    @Description
    A weighted MSE Loss. This loss function places a heavier weight on px and py
    values as well as calculated the pT for each particle and uses that as the
    weight. Expected input is (6 x 3). 6 particles. (px, py, pz) for each.
    ====================================================================
    @Input Format
    "target_W_had_Px", "target_W_had_Py", "target_W_had_Pz",
    "target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz",
    "target_b_had_Px", "target_b_had_Py", "target_b_had_Pz",
    "target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz",
    "target_t_had_Px", "target_t_had_Py", "target_t_had_Pz",
    "target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz"
    ====================================================================
    """
    loss = tf.constant(0.0)
    loss += tf.math.reduce_sum(tf.math.squared_difference(y_true[:2],y_pred[:2]))
    loss += 5 * tf.math.reduce_sum(tf.math.squared_difference(y_true[2:4,:2],y_pred[2:4,:2]))
    loss += tf.math.reduce_sum(tf.math.squared_difference(y_true[2:4,2],y_pred[2:4,2]))
    loss += tf.math.reduce_sum(tf.math.squared_difference(y_true[2:],y_pred[2:]))
    true_pT = tf.math.sqrt(tf.math.add(tf.math.square(y_true[:,0]), tf.math.square(y_true[:,1])))
    pred_pT = tf.math.sqrt(tf.math.add(tf.math.square(y_pred[:,0]), tf.math.square(y_pred[:,1])))
    loss += tf.math.reduce_sum(tf.math.squared_difference(true_pT,pred_pT))
    return loss / (1 * 20 + 5 * 4)

def pT_loss(y_true, y_pred):
    """
    @Description
    A weighted MSE Loss. This loss function places a heavier weight on px and py
    values as well as calculated the pT for each particle and uses that as the
    weight. Expected input is (6 x 3). 6 particles. (px, py, pz) for each.
    ====================================================================
    @Input Format
    "target_W_had_Px", "target_W_had_Py", "target_W_had_Pz",
    "target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz",
    "target_b_had_Px", "target_b_had_Py", "target_b_had_Pz",
    "target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz",
    "target_t_had_Px", "target_t_had_Py", "target_t_had_Pz",
    "target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz"
    ====================================================================
    """
    loss = tf.constant(0.0)
    true_pT = tf.math.sqrt(tf.math.add(tf.math.square(y_true[:,0]), tf.math.square(y_true[:,1])))
    pred_pT = tf.math.sqrt(tf.math.add(tf.math.square(y_pred[:,0]), tf.math.square(y_pred[:,1])))
    loss += tf.math.reduce_sum(tf.math.squared_difference(true_pT,pred_pT))
    return loss / 6

def w_HAD(y_true, y_pred):
    """
    @Description
    A weighted MSE Loss. This loss function places a heavier weight on px and py
    values as well as calculated the pT for each particle and uses that as the
    weight. Expected input is (6 x 3). 6 particles. (px, py, pz) for each.
    ====================================================================
    @Input Format
    "target_W_had_Px", "target_W_had_Py", "target_W_had_Pz",
    "target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz",
    "target_b_had_Px", "target_b_had_Py", "target_b_had_Pz",
    "target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz",
    "target_t_had_Px", "target_t_had_Py", "target_t_had_Pz",
    "target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz"
    ====================================================================
    """
    return tf.math.reduce_mean(tf.math.squared_difference(y_true[0],y_pred[0]))

def w_LEP(y_true, y_pred):
    """
    @Description
    A weighted MSE Loss. This loss function places a heavier weight on px and py
    values as well as calculated the pT for each particle and uses that as the
    weight. Expected input is (6 x 3). 6 particles. (px, py, pz) for each.
    ====================================================================
    @Input Format
    "target_W_had_Px", "target_W_had_Py", "target_W_had_Pz",
    "target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz",
    "target_b_had_Px", "target_b_had_Py", "target_b_had_Pz",
    "target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz",
    "target_t_had_Px", "target_t_had_Py", "target_t_had_Pz",
    "target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz"
    ====================================================================
    """
    return tf.math.reduce_mean(tf.math.squared_difference(y_true[1],y_pred[1]))

def b_HAD(y_true, y_pred):
    """
    @Description
    A weighted MSE Loss. This loss function places a heavier weight on px and py
    values as well as calculated the pT for each particle and uses that as the
    weight. Expected input is (6 x 3). 6 particles. (px, py, pz) for each.
    ====================================================================
    @Input Format
    "target_W_had_Px", "target_W_had_Py", "target_W_had_Pz",
    "target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz",
    "target_b_had_Px", "target_b_had_Py", "target_b_had_Pz",
    "target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz",
    "target_t_had_Px", "target_t_had_Py", "target_t_had_Pz",
    "target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz"
    ====================================================================
    """
    return tf.math.reduce_mean(tf.math.squared_difference(y_true[2],y_pred[2]))

def b_LEP(y_true, y_pred):
    """
    @Description
    A weighted MSE Loss. This loss function places a heavier weight on px and py
    values as well as calculated the pT for each particle and uses that as the
    weight. Expected input is (6 x 3). 6 particles. (px, py, pz) for each.
    ====================================================================
    @Input Format
    "target_W_had_Px", "target_W_had_Py", "target_W_had_Pz",
    "target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz",
    "target_b_had_Px", "target_b_had_Py", "target_b_had_Pz",
    "target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz",
    "target_t_had_Px", "target_t_had_Py", "target_t_had_Pz",
    "target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz"
    ====================================================================
    """
    return tf.math.reduce_mean(tf.math.squared_difference(y_true[3],y_pred[3]))


def t_HAD(y_true, y_pred):
    """
    @Description
    A weighted MSE Loss. This loss function places a heavier weight on px and py
    values as well as calculated the pT for each particle and uses that as the
    weight. Expected input is (6 x 3). 6 particles. (px, py, pz) for each.
    ====================================================================
    @Input Format
    "target_W_had_Px", "target_W_had_Py", "target_W_had_Pz",
    "target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz",
    "target_b_had_Px", "target_b_had_Py", "target_b_had_Pz",
    "target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz",
    "target_t_had_Px", "target_t_had_Py", "target_t_had_Pz",
    "target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz"
    ====================================================================
    """
    return tf.math.reduce_mean(tf.math.squared_difference(y_true[4],y_pred[4]))

def t_LEP(y_true, y_pred):
    """
    @Description
    A weighted MSE Loss. This loss function places a heavier weight on px and py
    values as well as calculated the pT for each particle and uses that as the
    weight. Expected input is (6 x 3). 6 particles. (px, py, pz) for each.
    ====================================================================
    @Input Format
    "target_W_had_Px", "target_W_had_Py", "target_W_had_Pz",
    "target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz",
    "target_b_had_Px", "target_b_had_Py", "target_b_had_Pz",
    "target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz",
    "target_t_had_Px", "target_t_had_Py", "target_t_had_Pz",
    "target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz"
    ====================================================================
    """
    return tf.math.reduce_mean(tf.math.squared_difference(y_true[4],y_pred[4]))
