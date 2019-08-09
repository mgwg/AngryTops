"""
This script contains custom loss functions and metrics
"""
import tensorflow as tf
import numpy as np

def weighted_MSE(weights):
    def weighted_mse(y_true, y_pred):
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
        loss += w_HAD(y_true, y_pred) * weights[0]
        loss += w_LEP(y_true, y_pred) * weights[1]
        loss += b_HAD(y_true, y_pred) * weights[2]
        loss += b_LEP(y_true, y_pred) * weights[3]
        loss += t_HAD(y_true, y_pred) * weights[4]
        loss += t_LEP(y_true, y_pred) * weights[5]
        return loss / sum(weights)
    return weighted_mse

def pTetaphi_Loss(y_true, y_pred):
    """
    @Description
    The mean squared error for the ptetaphi representation of the output.
    Assumed y_true and y_pred are both in the cartesian representation.
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
    pxpz_true = tf.reshape(tf.stack([y_true[:,0], y_pred[:,-1]], 1), shape=(-1,2))
    pxpz_pred = tf.reshape(tf.stack([y_true[:,0], y_pred[:,-1]], 1), shape=(-1,2))
    # PT contribution
    loss += tf.math.reduce_sum(tf.math.squared_difference(true_pT,pred_pT))
    # Eta contribution
    loss += tf.math.reduce_sum(tf.math.squared_difference(tf.math.angle(y_true[:,:2]), tf.math.angle(y_pred[:,:2])))
    # Phi contribution
    loss += tf.math.reduce_sum(tf.math.squared_difference(tf.math.angle(pxpz_true), tf.math.angle(pxpz_pred)))
    return loss / 18


def pT_loss(y_true, y_pred):
    """
    @Description
    PT loss. Expected input is (6 x 3). 6 particles. (px, py, pz) for each.
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

def transpose_subtraction(x, y):
    """
    a: tf tensor of shape (6, 3)
    b: tf tensor of shape(6, 3)
    Return: That dist array used in gaussian kernel
    """
    norm = lambda z: tf.reduce_sum(tf.square(z), 1)
    output_list = []
    for i in range(32):
        a = x[i]
        b = y[i]
        output_list.append(tf.transpose(norm(tf.expand_dims(a, len(a.shape)) - tf.transpose(b, perm=[1,0]))))
    dist = tf.stack(output_list)
    return dist

def gaussian_kernel(x,y):
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    sigmas = np.ones(36) * 1.
    sigmas = tf.constant(sigmas)
    dist = transpose_subtraction(x, y)
    dist = tf.dtypes.cast(dist, dtype=tf.float64)
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def mmd_loss(y_true, y_pred):
    """MMD^2(P, Q) = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)> is a radial basis kernel (gaussian)
    """
    cost = tf.reduce_mean(gaussian_kernel(y_true, y_true))
    cost += tf.reduce_mean(gaussian_kernel(y_pred, y_pred))
    cost -= 2 * tf.reduce_mean(gaussian_kernel(y_true, y_pred))
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost

def mmd_mse_loss(y_true, y_pred):
    """Sum of mmd loss and mse loss. Note: Somewhat buggy."""
    x = tf.dtypes.cast(mmd_loss(y_true, y_pred), dtype=tf.float64)
    y = tf.dtypes.cast(weighted_MSE([1,1,1,1,1,1])(y_true, y_pred), dtype=tf.float64)
    return x * 0.1 + y * 0.9

###############################################################################
# Default metrics + losses. The weighted metrics get added later
custom_metrics={"w_HAD":w_HAD, "w_LEP":w_LEP, "b_HAD":b_HAD, "b_LEP":b_LEP,
                "t_HAD":t_HAD, "t_LEP":t_LEP, "pT_loss":pT_loss,
                'pTetaphi_Loss':pTetaphi_Loss, "mmd_loss":mmd_loss,
                'mmd_mse_loss':mmd_mse_loss}
metrics = ['mae', 'mse']
losses = {"mse":"mse", "pT_loss":pT_loss, "pTetaphi_Loss":pTetaphi_Loss,
          "mmd_loss":mmd_loss, "mmd_mse_loss":mmd_mse_loss}

if __name__ == "__main__":
    x = np.random.randint(1, 10, 18*32).reshape(32, 6, 3) * 1.
    y = np.random.randint(1, 10, 18*32).reshape(32 ,6, 3) * 1.
    print(mmd_loss(x, y))
