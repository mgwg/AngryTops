"""
This script contains custom loss functions and metrics
"""
import tensorflow as tf

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
        return loss
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



# def maximum_mean_discrepancy(x, y, kernel=utils.gaussian_kernel_matrix):
#   r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
#   Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
#   the distributions of x and y. Here we use the kernel two sample estimate
#   using the empirical mean of the two distributions.
#   MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
#               = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
#   where K = <\phi(x), \phi(y)>,
#     is the desired kernel function, in this case a radial basis kernel.
#   Args:
#       x: a tensor of shape [num_samples, num_features]
#       y: a tensor of shape [num_samples, num_features]
#       kernel: a function which computes the kernel in MMD. Defaults to the
#               GaussianKernelMatrix.
#   Returns:
#       a scalar denoting the squared maximum mean discrepancy loss.
#   """
#   with tf.name_scope('MaximumMeanDiscrepancy'):
#     # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
#     cost = tf.reduce_mean(kernel(x, x))
#     cost += tf.reduce_mean(kernel(y, y))
#     cost -= 2 * tf.reduce_mean(kernel(x, y))
#
#     # We do not allow the loss to become negative.
#     cost = tf.where(cost > 0, cost, 0, name='value')
#   return cost

#
# def mmd_loss(source_samples, target_samples, weight, scope=None):
#   """Adds a similarity loss term, the MMD between two representations.
#   This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
#   different Gaussian kernels.
#   Args:
#     source_samples: a tensor of shape [num_samples, num_features].
#     target_samples: a tensor of shape [num_samples, num_features].
#     weight: the weight of the MMD loss.
#     scope: optional name scope for summary tags.
#   Returns:
#     a scalar tensor representing the MMD loss value.
#   """
#   sigmas = [
#       1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
#       1e3, 1e4, 1e5, 1e6
#   ]
#   gaussian_kernel = partial(
#       utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
#
#   loss_value = maximum_mean_discrepancy(
#       source_samples, target_samples, kernel=gaussian_kernel)
#   loss_value = tf.maximum(1e-4, loss_value) * weight
#   assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
#   with tf.control_dependencies([assert_op]):
#     tag = 'MMD Loss'
#     if scope:
#       tag = scope + tag
#     tf.summary.scalar(tag, loss_value)
#     tf.losses.add_loss(loss_value)
#
#   return loss_value
#
#
# def gaussian_kernel(x,y):
#     norm = lambda x: tf.reduce_sum(tf.square(x), 1)
#     sigmas = [ 1., 1., 1., 1., 1., 1., 1. ]
#     sigmas = tf.constant(sigmas)
#     print(tf.square(x).shape)
#     print(y.shape)
#     dist = tf.transpose(norm(x - tf.transpose(y)))
#     beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
#     s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
#     return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

#def mmd_loss(y_true, y_pred):
#    """MMD^2(P, Q) = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
#    where K = <\phi(x), \phi(y)> is a radial basis kernel (gaussian)
#    """
#    cost = tf.reduce_mean(gaussian_kernel(y_true, y_true))
#    cost += tf.reduce_mean(gaussian_kernel(y_pred, y_pred))
#    cost -= 2 * tf.reduce_mean(gaussian_kernel(y_true, y_pred))
#    cost = tf.where(cost > 0, cost, 0, name='value')
#    return cost

###############################################################################
# Default metrics + losses. The weighted metrics get added later
custom_metrics={"w_HAD":w_HAD, "w_LEP":w_LEP, "b_HAD":b_HAD, "b_LEP":b_LEP,
                "t_HAD":t_HAD, "t_LEP":t_LEP, "pT_loss":pT_loss,
                'pTetaphi_Loss':pTetaphi_Loss}
metrics = ['mae', 'mse', w_HAD, w_LEP, b_HAD, b_LEP, t_HAD, t_LEP, pT_loss,
            pTetaphi_Loss]
losses = {"mse":"mse", "pT_loss":pT_loss, "pTetaphi_Loss":pTetaphi_Loss}

if __name__ == "__main__":
    a = np.random.randint(low=1, high=100, size=32*6*3).reshape(32,6,3)
    b = np.random.randint(low=1, high=100, size=32*6*3).reshape(32,6,3)
    print(mmd_loss(a, b))
