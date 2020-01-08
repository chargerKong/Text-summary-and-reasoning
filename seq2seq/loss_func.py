import tensorflow as tf


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred, mask):
    # 　计算交叉熵
    import pdb
    pdb.set_trace()
    loss_ = loss_object(real, pred)
    # 吧mask转换为loss的形式
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def totally_loss(real, pred, padding_mask, attention_dist, cov_loss_wt):

    import pdb
    pdb.set_trace()
    log_loss = loss_function(real, pred, padding_mask)
    coverage_loss = _coverage_loss(attn_dists=attention_dist, padding_mask=padding_mask)

    return log_loss + cov_loss_wt * coverage_loss, log_loss, coverage_loss


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
      attn_dists: The attention distributions
      for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).

    Returns:
      coverage_loss: scalar
    """
    coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
      a scalar
    """
    padding_mask = tf.cast(padding_mask, dtype=values[0].dtype)
    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex)  # overall average
