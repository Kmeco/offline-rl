import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_probability as tfp

import sonnet as snt

tfd = tfp.distributions


class GreedyHead(snt.Module):
  """Simple sonnet module to sample from a tfp.Distribution."""

  def __call__(self, distribution: tfd.Distribution):
    return tf.argmax(distribution.logits, axis=1)


def get_default_critic(env_spec):
  critic = snt.Sequential([
    snt.Flatten(),
    snt.nets.MLP([128, 64, 32, env_spec.actions.num_values]),
  ])
  return critic