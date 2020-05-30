import tensorflow as tf
from  tensorflow_probability import distributions as tfd


@tf.function
def loss_elbo(model, x, **call_args):
    model_outputs = model(x, **call_args)
    kl_regulariser = tf.reduce_mean(tf.reduce_sum(model.kl_loss(model_outputs), axis=1))
    log_probabilities = model.decoder_dist(
        loc=model_outputs['decoded'], eps_scale=model.decoder_eps_scale).log_prob(x)
    likelihood = tf.reduce_mean(tf.reduce_sum(log_probabilities, axis=[1, 2, 3]))
    negative_elbo = -likelihood + kl_regulariser 
    return dict(final=negative_elbo, likelihood=likelihood, kl=kl_regulariser)

@tf.function
def loss_elbo_cross_ent(model, x, cross_ent='sigmoid', **call_args):
    model_outputs = model(x, **call_args)
    kl_regulariser = tf.reduce_mean(tf.reduce_sum(model.kl_loss(model_outputs), axis=1))
    if cross_ent=='sigmoid':
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=model_outputs['decoded'], labels=x)
    else: 
        cross_ent = tf.nn.softmax_cross_entropy_with_logits(
            logits=model_outputs['decoded'], labels=x)
    likelihood = -tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))
    negative_elbo = -likelihood + kl_regulariser 
    return dict(final=negative_elbo, likelihood=likelihood, kl=kl_regulariser)

@tf.function
def compute_apply_gradients(model, x, compute_loss, optimizer, **call_args):
    with tf.GradientTape() as tape:
        loss_dict = compute_loss(model, x, **call_args) 
        loss = loss_dict['final']
        loss = tf.debugging.check_numerics(loss, 'nan in loss')
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_dict