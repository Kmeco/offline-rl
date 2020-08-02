import operator
import os
import time

import dm_env
import tree
import reverb
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
from typing import Any, List

from acme import core, specs, types


def n_step_transition_from_episode(observations: types.NestedTensor,
                                    actions: tf.Tensor, rewards: tf.Tensor,
                                    discounts: tf.Tensor, n_step: int,
                                    additional_discount: float):
    """Produce Reverb-like N-step transition from a full episode.

    Observations, actions, rewards and discounts have the same length. This
    function will ignore the first reward and discount and the last action.

    Args:
    observations: [L, ...] Tensor.
    actions: [L, ...] Tensor.
    rewards: [L] Tensor.
    discounts: [L] Tensor.
    n_step: number of steps to squash into a single transition.
    additional_discount: discount to use for TD updates.

    Returns:
    (o_t, a_t, r_t, d_t, o_tp1) tuple.
    """

    max_index = tf.shape(rewards)[0] - 1
    first = tf.random.uniform(
      shape=(), minval=0, maxval=max_index - 1, dtype=tf.int32)
    last = tf.minimum(first + n_step, max_index)

    o_t = tree.map_structure(operator.itemgetter(first), observations)
    a_t = tree.map_structure(operator.itemgetter(first), actions)
    o_tp1 = tree.map_structure(operator.itemgetter(last), observations)

    # 0, 1, ..., n-1.
    discount_range = tf.cast(tf.range(last - first), tf.float32)
    # 1, g, ..., g^{n-1}.
    additional_discounts = tf.pow(additional_discount, discount_range)
    # 1, d_t, d_t * d_{t+1}, ..., d_t * ... * d_{t+n-2}.
    discounts = tf.concat([[1.], tf.math.cumprod(discounts[first:last - 1])], 0)
    # 1, g * d_t, ..., g^{n-1} * d_t * ... * d_{t+n-2}.
    discounts *= additional_discounts
    # r_t + g * d_t * r_{t+1} + ... + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}
    # We have to shift rewards by one so last=max_index corresponds to transitions
    # that include the last reward.
    r_t = tf.reduce_sum(rewards[first + 1:last + 1] * discounts)

    # g^{n-1} * d_{t} * ... * d_{t+n-1}.
    d_t = discounts[-1]

    # Reverb requires every sample to be given a key and priority.
    # In the supervised learning case for BC, neither of those will be used.
    # We set the key to `0` and the priorities probabilities to `1`, but that
    # should not matter much.
    key = tf.constant(0, tf.uint64)
    probability = tf.constant(1.0, tf.float64)
    table_size = tf.constant(1, tf.int64)
    priority = tf.constant(1.0, tf.float64)
    info = reverb.SampleInfo(
      key=key,
      probability=probability,
      table_size=table_size,
      priority=priority,
    )

    return reverb.ReplaySample(info=info, data=(o_t, a_t, r_t, d_t, o_tp1))


class DemonstrationRecorder:
    def __init__(self, env, agent):
        self._episodes = []
        self._ep_buffer = []
        self.env = env
        self._env_spec = specs.make_environment_spec(env)
        self.agent = agent
        self.empirical_policy = {}
        self._prev_observation = None

    def collect_episode(self):
        """ collects tuples:
                o_t: Observation at time t.
                a_t: Action at time t.
                r_t: Reward at time t.
                d_t: Discount at time t.
                extras: Dictionary with extra features."""
        self._ep_buffer = []
        timestep = self.env.reset()
        self._prev_observation = timestep.observation

        while not timestep.last():
            action = self.agent.select_action(timestep.observation)
            timestep = self.env.step(action)
            self._step(timestep, action)
            self._update_policy_counts(self._prev_observation, action)
            self._prev_observation = timestep.observation

        self._episodes.append(_nested_stack(self._ep_buffer))

    def collect_n_episodes(self, n):
        for _ in tqdm(range(n)):
            self.collect_episode()

    def _update_policy_counts(self, observation, action):
        if self.empirical_policy.get(str(observation)) is not None:
            self.empirical_policy[str(observation)][action] += 1
        else:
            self.empirical_policy[str(observation)] = np.zeros(self._env_spec.actions.num_values)

    def _step(self, timestep, action):
        self._ep_buffer.append((self._prev_observation,
                                action,
                                timestep.reward,
                                timestep.discount))

    def make_tf_dataset(self):
        self.types = tree.map_structure(lambda x: x.dtype, self._episodes[0])
        # the shapes are given by None since the ep length varies
        self.shapes = ((None, self._episodes[0][0].shape[1]), (None,), (None,), (None,))

        self.ds = tf.data.Dataset.from_generator(lambda: self._episodes, self.types, self.shapes)
        # .repeat().shuffle(len(dr._episodes))
        return self.ds

    @tf.autograph.experimental.do_not_convert
    def save(self, directory='datasets', overwrite=False):
        if not overwrite:
            directory = os.path.join(directory, str(int(time.time())))
        os.makedirs(directory, exist_ok=True)

        spec = {'types': self.types,
                'shapes': self.shapes,
                'policy': self.empirical_policy}

        with open('spec.pkl', 'wb') as f:
            pickle.dump(spec, f)

        for i, _ in enumerate(self.ds.element_spec):
            file_path = os.path.join(directory, f'offline_data.{i}.tfrecord')
            ds_i = self.ds.map(lambda *args: args[i]).map(tf.io.serialize_tensor)
            writer = tf.data.experimental.TFRecordWriter(file_path, compression_type='GZIP')
            writer.write(ds_i)


@tf.autograph.experimental.do_not_convert
def load_tf_dataset(directory='datasets'):

    spec_path = os.path.join(directory, 'spec.pkl')
    with open(spec_path, 'rb') as f:
        spec = pickle.load(f)

    parts = []
    for i, dtype in enumerate(spec['types']):
        file_path = os.path.join(directory, f'offline_data.{i}.tfrecord')
        parts.append(tf.data.TFRecordDataset(file_path, compression_type='GZIP').map(
                                             lambda x: tf.io.parse_tensor(x, dtype)))
    return tf.data.Dataset.zip(tuple(parts)), spec['policy']


def _nested_stack(sequence: List[Any]):
    """Stack nested elements in a sequence."""
    return tree.map_structure(lambda *x: np.stack(x), *sequence)


class RandomActor(core.Actor):
  """Fake actor which generates random actions and validates specs."""

  def __init__(self, spec: specs.EnvironmentSpec):
    self._spec = spec
    self.num_updates = 0

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    _validate_spec(self._spec.observations, observation)
    return np.random.randint(self._spec.actions.num_values)

  def observe_first(self, timestep: dm_env.TimeStep):
    _validate_spec(self._spec.observations, timestep.observation)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    _validate_spec(self._spec.actions, action)
    _validate_spec(self._spec.rewards, next_timestep.reward)
    _validate_spec(self._spec.discounts, next_timestep.discount)
    _validate_spec(self._spec.observations, next_timestep.observation)

  def update(self):
    self.num_updates += 1

def _validate_spec(spec: types.NestedSpec, value: types.NestedArray):
  """Validate a value from a potentially nested spec."""
  tree.assert_same_structure(value, spec)
  tree.map_structure(lambda s, v: s.validate(v), spec, value)