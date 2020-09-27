import tensorflow as tf
import numpy as np
from cleverhans.compat import reduce_max, softmax_cross_entropy_with_logits, reduce_sum


def gaussian_kernel(size: int, mean: float, std: float, ):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum(
        'i,j->ij',
        vals,
        vals
    )
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def total_v(p, shape):
    d = tf.reshape(p, shape=shape)  # (NI, h, w, 3)

    n_inputs = shape[0]
    n_rows = shape[1]
    n_cols = shape[2]
    n_channels = shape[3]

    v1, v2 = d[:, -1:], d[:, 1:]  # (NI, h-1, w, 3)
    h1, h2 = d[:, :, -1:], d[:, :, 1:]  # (NI, h, w-1, 3)
    zero_row = tf.zeros(shape=(n_inputs, 1, n_cols, n_channels), dtype=tf.float32)
    zero_column = tf.zeros(shape=(n_inputs, n_rows, 1, n_channels), dtype=tf.float32)

    ddown = tf.concat((tf.abs(v1 - v2), zero_row), axis=1)
    dup = tf.concat((zero_row, tf.abs(v2 - v1)), axis=1)
    dright = tf.concat((tf.abs(h1 - h2), zero_column), axis=2)
    dleft = tf.concat((zero_column, tf.abs(h2 - h1)), axis=2)

    map = ddown + dup + dright + dleft
    map = tf.sqrt(tf.reduce_max(map) + 1.0 - map)

    return map


def roll_glasses(g, offsets_x, offsets_y):
    """

    :param g: tensor of shape (?, height, width, channels)
    :param offsets_x: tensor of shape (?, ) that contains the shift on the width axis
    :param offsets_y: tensor of shape (?, ) that contains the shift on the height axis
    :return: g but with the rolling shift
    """
    res1, _ = tf.map_fn(
        lambda x: (tf.manip.roll(x[0], x[1], axis=1), 0),
        (g, offsets_x),
        dtype=(tf.float32, tf.int32)
    )
    res2, _ = tf.map_fn(
        lambda x: (tf.manip.roll(x[0], x[1], axis=0), 0),
        (res1, offsets_y),
        dtype=(tf.float32, tf.int32)
    )
    return tf.identity(res2)


def fix_domain(pfs_in, dp, dm, gmask):
    # remove stuff outside of domain
    pfs_out = tf.multiply(pfs_in, gmask)  # (NI, NF)
    pfs_p = tf.multiply(pfs_out, dp)  # (NI, NF)
    pfs_m = tf.multiply(pfs_out, dm)  # (NI, NF)
    return pfs_p, pfs_m


def get_changes_from_gradients(pfs_p_in, pfs_m_in, n_features, n_samples, n_pixels):
    """

    :param pfs_p_in: tensor (?, n_features)
    :param pfs_m_in: tensor (?, n_features)
    :param n_features:
    :param n_samples:
    :param n_pixels:
    :return:
    """

    width = height = np.sqrt(n_features/3).astype(int)

    pfs_p_out = tf.reduce_sum(pfs_p_in, axis=0, )  # (NF, )
    pfs_m_out = tf.reduce_sum(pfs_m_in, axis=0, )  # (NF, )

    _, best_p = tf.nn.top_k(pfs_p_out, k=n_features, sorted=True)
    _, best_m = tf.nn.top_k(pfs_m_out, k=n_features, sorted=True)

    best_p = best_p[:n_pixels]
    best_m = tf.reverse(best_m, axis=[0])[:n_pixels]

    #best_p = tf.print(best_p, [best_p], message="+,", summarize=n_pixels)
    #best_m = tf.print(best_m, [best_m], message="-,", summarize=n_pixels)

    one_hot_p = tf.reduce_sum(tf.one_hot(best_p, depth=n_features, dtype=tf.float32), axis=0)  # (NF, )
    one_hot_m = tf.reduce_sum(tf.one_hot(best_m, depth=n_features, dtype=tf.float32), axis=0)  # (NF, )
    one_hot_p = tf.tile(one_hot_p, [n_samples])  # (NIxNF, )
    one_hot_m = tf.tile(one_hot_m, [n_samples])  # (NIxNF, )
    one_hot_p = tf.reshape(one_hot_p, [-1, height, width, 3])  # (NI, h, w, 3)
    one_hot_m = tf.reshape(one_hot_m, [-1, height, width, 3])  # (NI, h, w, 3)
    return one_hot_p, one_hot_m, best_p, best_m


def fgsm_m(model, theta, clip_min, clip_max, glasses_mask, offsets, n_pixels, i_shape):
    # do some checks
    # assert model.output_shape == y_target.shape[-1].value
    assert glasses_mask.dtype == tf.bool

    x = model.face_input
    preds = model.get_probs(x)
    logits, = preds.op.inputs
    #print("L", logits)
    # logits = model.get_logits(x)[0]
    print("[INFO] - Softmax Layer", preds)
    print("[INFO] - Logits Layer", logits)

    clip_min = tf.cast(clip_min, tf.float32)
    clip_max = tf.cast(clip_max, tf.float32)

    n_samples = glasses_mask.shape[0].value
    n_features = int(np.product(glasses_mask.shape[1:]).value)

    glasses_mask = tf.cast(glasses_mask, dtype=tf.float32)
    search_domain_plus = tf.reshape(tf.cast(x < clip_max, tf.float32), [-1, n_features]) * glasses_mask
    search_domain_minus = tf.reshape(tf.cast(x > clip_min, tf.float32), [-1, n_features]) * glasses_mask

    preds_max = reduce_max(logits, 1, keepdims=True)
    y = tf.to_float(tf.equal(logits, preds_max))
    y = tf.stop_gradient(y)

    y = y / reduce_sum(y, 1, keepdims=True)
    # Compute loss
    mean, var = tf.nn.moments(logits, axes=[1])
    loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # Define gradient of loss wrt input
    grads, = tf.gradients(loss, x)  # (NI, w, h, 3)
    per_feature_score = tf.reshape(grads, shape=(-1, n_features))  # (NI, NF)

    # get glasses total variation to prioritize smoother changes
    glasses = tf.multiply(tf.reshape(x, shape=(-1, n_features)), glasses_mask)  # (NI, NF)
    tot_v = tf.reshape(total_v(glasses, shape=(n_samples,) + i_shape), shape=[-1, n_features])

    # fix domain
    pfs_p, pfs_m = fix_domain(per_feature_score, search_domain_plus, search_domain_minus, glasses_mask)
    pfs_p = tf.multiply(pfs_p, tot_v)  # (NI, NF)
    pfs_m = tf.multiply(pfs_m, tot_v)  # (NI, NF)

    # only consider worst sample
    # pfs_p = tf.multiply(pfs_p, furthest)  # (NI x NF)
    # pfs_m = tf.multiply(pfs_m, furthest)  # (NI x NF)
    # per_feature_score = tf.multiply(per_feature_score, furthest)  # (NI, NF) x (NI, 1) = (NI, NF)

    # roll to top left
    pfs_p = tf.reshape(pfs_p, (-1,) + i_shape)
    pfs_m = tf.reshape(pfs_m, (-1,) + i_shape)
    pfs_p = roll_glasses(pfs_p, -offsets[:, 0], -offsets[:, 1])
    pfs_m = roll_glasses(pfs_m, -offsets[:, 0], -offsets[:, 1])

    # get changes
    pfs_p = tf.reshape(pfs_p, (-1, n_features))
    pfs_m = tf.reshape(pfs_m, (-1, n_features))
    one_hot_p, one_hot_m, best_p, best_m = get_changes_from_gradients(pfs_p, pfs_m, n_features, n_samples, n_pixels)  # (NI, h, w, 3)

    # roll back to actual position
    one_hot_p = roll_glasses(one_hot_p, offsets[:, 0], offsets[:, 1])
    one_hot_m = roll_glasses(one_hot_m, offsets[:, 0], offsets[:, 1])

    # apply changes
    x_out = x + one_hot_p * theta
    x_out = tf.minimum(clip_max, x_out)
    x_out = x_out - one_hot_m * theta
    x_out = tf.maximum(clip_min, x_out)

    return x_out, best_p, best_m, loss, var


def fgsm(x, y_target, model, theta, clip_min, clip_max, glasses_mask, offsets, n_pixels):
    """
    :param x: placeholder for input, shape (?, ?, ?, ?)
    :param y_target: target centroids, shape (n_classes, emb_dimension)
    :param model: subclassing BaseRecognizer. Needs to implement fprop
    :param theta: regulates amount of distortion
    :param clip_min:
    :param clip_max:
    :param glasses_mask: boolean array with True for the glasses location, shape (n, height*width*channels)
    :param maxiters:
    :param glasses: array containing the glasses, shape (n, height, width, channels)
    :return:
    """

    # do some checks
    # assert model.output_shape == y_target.shape[-1].value
    assert glasses_mask.dtype == tf.bool

    clip_min = tf.cast(clip_min, tf.float32)
    clip_max = tf.cast(clip_max, tf.float32)

    n_samples = y_target.shape[0].value
    n_classes = int(y_target.shape[-1].value)
    n_features = int(np.product(glasses_mask.shape[1:]).value)

    glasses_mask = tf.cast(glasses_mask, dtype=tf.float32)
    search_domain_plus = tf.reshape(tf.cast(x < clip_max, tf.float32), [-1, n_features]) * glasses_mask
    search_domain_minus = tf.reshape(tf.cast(x > clip_min, tf.float32), [-1, n_features]) * glasses_mask

    # start body
    list_derivatives = []
    preds = model.get_probs(x)  # (NI, NC)

    for class_ind in range(n_classes):
        print(class_ind)
        if class_ind > 1:
            list_derivatives.append(derivatives[0])
        else:
            derivatives = tf.gradients(preds[:, class_ind], x)  # (NI, width, height, channels)
            list_derivatives.append(derivatives[0])

    grads = tf.reshape(tf.stack(list_derivatives), shape=[n_classes, -1, n_features])  # (NC, NI, NF)
    grads = tf.transpose(grads, [1, 2, 0])  # (NI, NF, NC)

    distances = tf.subtract(y_target, preds)  # (NI, NC)
    directions = tf.expand_dims(distances, 1)  # (NI, 1, NC)

    # l2d = tf.norm(distances, ord=2, axis=1)  # (NI, )
    # furthest_away_sample = tf.argmax(l2d)
    # furthest_away_sample = tf.Print(furthest_away_sample, [furthest_away_sample], message="i,", summarize=1000)
    # furthest = tf.one_hot(furthest_away_sample, depth=n_samples)  # (NI, )
    # furthest = tf.expand_dims(furthest, 1)  # (NI, 1)
    # furthest = tf.Print(furthest, [furthest], message="i2,", summarize=1000)

    row_wise_dot_product = tf.multiply(grads, directions)  # (NI, NF, NC)
    per_feature_score = tf.reduce_mean(row_wise_dot_product, axis=-1)  # (NI, NF)

    # fix domain
    pfs_p, pfs_m = fix_domain(per_feature_score, search_domain_plus, search_domain_minus, glasses_mask)

    # only consider worst sample
    # pfs_p = tf.multiply(pfs_p, furthest)  # (NI x NF)
    # pfs_m = tf.multiply(pfs_m, furthest)  # (NI x NF)
    # per_feature_score = tf.multiply(per_feature_score, furthest)  # (NI, NF) x (NI, 1) = (NI, NF)

    # roll to top left
    pfs_p = tf.reshape(pfs_p, (-1, 160, 160, 3))
    pfs_m = tf.reshape(pfs_m, (-1, 160, 160, 3))
    pfs_p = roll_glasses(pfs_p, -offsets[:, 0], -offsets[:, 1])
    pfs_m = roll_glasses(pfs_m, -offsets[:, 0], -offsets[:, 1])

    # get changes
    pfs_p = tf.reshape(pfs_p, (-1, n_features))
    pfs_m = tf.reshape(pfs_m, (-1, n_features))
    one_hot_p, one_hot_m, best_p, best_m = get_changes_from_gradients(pfs_p, pfs_m, n_features, n_samples,
                                                                      n_pixels)  # (NI, 160, 160, 3)

    # roll back to actual position
    one_hot_p = roll_glasses(one_hot_p, offsets[:, 0], offsets[:, 1])
    one_hot_m = roll_glasses(one_hot_m, offsets[:, 0], offsets[:, 1])

    # apply changes
    x_out = x + one_hot_p * theta
    x_out = tf.minimum(clip_max, x_out)
    x_out = x_out - one_hot_m * theta
    x_out = tf.maximum(clip_min, x_out)

    return x_out, best_p, best_m
