import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
import math


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """
    force gradient to be a dense tensor
    it's often faster to do dense embedding gradient on GPU than sparse on CPU
    """
    return x


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def norm(x, scope, axis=[-1], e=1e-5):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + e)
        x = x * g + b
        return x


def dropout(x, pdrop):
    if pdrop > 0:
        x = tf.nn.dropout(x, 1 - pdrop)
    return x


def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w * b + -1e9 * (1 - b)
    return w


def _attn(q, k, v, sequence_lens, pdrop, mask):
    w = tf.matmul(q, k)
    shape = shape_list(v)
    n_state = shape[-1]
    seq_len = shape[-2]
    w = w * tf.rsqrt(tf.cast(n_state, tf.float32))

    if sequence_lens is not None:
        w *= tf.expand_dims(tf.expand_dims(tf.sequence_mask(sequence_lens, maxlen=seq_len, dtype=tf.float32),
                                           axis=1), axis=1)
    if mask:
        w = mask_attn_weights(w)

    w = tf.nn.softmax(w)
    w = dropout(w, pdrop)
    a = tf.matmul(w, v)
    return a


def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return tf.reshape(x, new_x_shape)


def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))


def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0),
           pad='VALID', train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1:  # faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b, shape_list(x)[:-1] + [nf])
        else:  # was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad) + b
    return c


def attn(x, scope, n_state, n_head, pdrop, sequence_lens, mask):
    assert n_state % n_head == 0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state * 3, 1)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, sequence_lens, pdrop=pdrop, mask=mask)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1)
        a = dropout(a, pdrop)
    return a


def cond_attn(x, cond, scope, n_state, n_head, pdrop, sequence_lens, mask):
    assert n_state % n_head == 0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state * 2, 1)
        q, k, v = tf.split(c, 2, 2)
        q = split_heads(q, n_head)
        v = split_heads(v, n_head)

        c = conv1d(tf.expand_dims(cond, dim=1), 'c_attn', n_state * 1, 1)
        k = split_heads(c, n_head, k=True)

        a = _attn(q, k, v, sequence_lens, pdrop=pdrop, mask=mask)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1)
        a = dropout(a, pdrop)
    return a


def decoder(x, targets, cond, embeddings, bias, n_head, n_block, pdrop, sequence_lens, vocab_size, num_sampled):
    """

    :param x: embedded input
    :param targets: target ids
    :param cond:
    :param embeddings:
    :param bias:
    :param n_head:
    :param n_block:
    :param pdrop:
    :param sequence_lens:
    :return:
    """
    targets = targets[:, 1:]
    x = add_timing_signal_1d(x[:, :-1, :])

    for i in range(n_block):
        x = decoder_block(x, cond, n_head, None, pdrop, "decoder_block_{}".format(i))

    shape = shape_list(x)
    seq_len = shape[1]

    loss_mask = tf.reshape(tf.sequence_mask(sequence_lens, maxlen=seq_len, dtype=tf.float32), shape=[-1])
    loss = tf.nn.sampled_softmax_loss(
        tf.reshape(embeddings, shape=[-1, shape[-1]]),
        bias,
        tf.reshape(targets, shape=[-1]),
        x,
        num_sampled,
        vocab_size,
        num_true=1,
        remove_accidental_hits=True,
        partition_strategy='div',
        name='reconstruction_loss',
    )
    loss = (loss_mask * loss) / tf.reduce_sum(loss_mask)
    return loss


def mlp(x, scope, n_state, pdrop, nx=None):
    with tf.variable_scope(scope):
        nx = nx or shape_list(x)[-1]
        h = tf.nn.relu(conv1d(x, 'c_fc', n_state, 1))
        h2 = conv1d(h, 'c_proj', nx, 1)
        h2 = dropout(h2, pdrop)
    return h2


def block(x, n_head, sequence_lens, pdrop, scope):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, pdrop, sequence_lens, mask=False)
        n = norm(x + a, 'ln_1')
        m = mlp(n, 'mlp', nx * 4, pdrop)
        h = norm(n + m, 'ln_2')
    return h


def decoder_block(x, cond, n_head, sequence_lens, pdrop, scope):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, n_head, pdrop, sequence_lens, mask=True)
        n = norm(x + a, 'ln_1')
        a = cond_attn(n, 'attn', nx, cond, n_head, pdrop, sequence_lens, mask=True)
        n = norm(n + a, 'ln_2')
        m = mlp(n, 'mlp', nx * 4, pdrop)
        h = norm(n + m, 'ln_3')
    return h


def add_timing_signal_1d(x,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    length = shape_list(x)[1]
    channels = shape_list(x)[2]
    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal + x
