import operator
import os
import random
import time
import wandb

import gym
import dm_env
import tree
import reverb
import tensorflow as tf
import numpy as np
import pickle

from absl import logging
from acme.utils import loggers
from acme.wrappers import gym_wrapper
from tqdm import tqdm
from typing import Any, List

from acme import core, specs, types

from acme.utils.loggers import base

from custom_env_wrappers import CustomSinglePrecisionWrapper, ImgFlatObsWrapper
from gym_minigrid.wrappers import FullyObsWrapper

WANDB_PROJECT_PATH = 'kmeco/offline-rl/{}:{}'


def _build_custom_loggers(wb_client):
  terminal_learner = loggers.TerminalLogger(label='Learner', time_delta=10)
  terminal_eval = loggers.TerminalLogger(label='EvalLoop', time_delta=10)

  if wb_client is not None:
    wb_learner = WBLogger(wb_client, label='Learner')
    wb_loop = WBLogger(wb_client, label='EvalLoop')
    disp = loggers.Dispatcher([terminal_learner, wb_learner])
    disp_loop = loggers.Dispatcher([terminal_eval, wb_loop])
    return disp, disp_loop
  else:
    return terminal_learner, terminal_eval


def _build_environment(name, n_actions=3, max_steps=500):
  raw_env = gym.make(name)
  raw_env.action_space.n = n_actions
  raw_env.max_steps = max_steps
  env = ImgFlatObsWrapper(FullyObsWrapper(raw_env))
  env = gym_wrapper.GymWrapper(env)
  env = CustomSinglePrecisionWrapper(env)
  spec = specs.make_environment_spec(env)
  return env, spec


class WBLogger(base.Logger):
  """Logs to wandb.

  If multiple TFSummaryLogger are created with the same logdir, results will be
  categorized by labels.
  """

  def __init__(
    self,
    wb_run,
    label: str = '',
  ):
    """Initializes the logger.

    Args:
      logdir: directory to which we should log files.
      label: label string to use when logging. Default to 'Logs'.
    """
    self._time = time.time()
    self.label = label + '/'
    self._iter = 0
    self._wandb = wb_run

  def write(self, values: base.LoggingData):
    self._wandb.log({self.label + k: v for k, v in values.items()})
    self._iter += 1


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
    shape=(), minval=0, maxval=max_index, dtype=tf.int32)
  last = tf.minimum(first + n_step, max_index)

  o_t = tree.map_structure(operator.itemgetter(first), observations)
  a_t = tree.map_structure(operator.itemgetter(first), actions)
  o_tp1 = tree.map_structure(operator.itemgetter(last), observations)

  # 0, 1, ..., n-1.
  discount_range = tf.cast(tf.range(last - first), tf.float32)
  # 1, g, ..., g^{n-1}.
  additional_discounts = tf.pow(additional_discount, discount_range)
  # 1, d_t, d_t * d_{t+1}, ..., d_t * ... * d_{t+n-2}.
  d_t = discounts[last - 1] * additional_discount ** tf.cast((last - first), tf.float32)
  discounts = tf.concat([[1.], tf.math.cumprod(discounts[first:last - 1])], 0)
  # 1, g * d_t, ..., g^{n-1} * d_t * ... * d_{t+n-2}.
  discounts *= additional_discounts
  #  r_t + g * d_t * r_{t+1} + ... + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}
  # We have to shift rewards by one so last=max_index corresponds to transitions
  # that include the last reward.
  r_t = tf.reduce_sum(rewards[first:last] * discounts)

  # g^{n-1} * d_{t} * ... * d_{t+n-1}.
  # d_t = discounts[-1]

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
  def __init__(self, env, agent, subsample=0):
    """
    Recorder that uses an agent to collect demonstrations
    by interacting with the given environment.
    :param subsample: percentage of zero reward trajectories to keep
    """
    self._episodes = []
    self._ep_buffer = []
    self.env = env
    self._env_spec = specs.make_environment_spec(env)
    self.agent = agent
    self._prev_observation = None
    self._subsample = subsample

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
      action = self.agent.select_action(self._prev_observation)
      timestep = self.env.step(action)
      self._record_step(timestep, action)
      self._prev_observation = timestep.observation

    self._record_step(timestep, np.zeros_like(action))
    if not self._subsample or timestep.reward:
      self._episodes.append(_nested_stack(self._ep_buffer))
      return True
    elif random.random() < self._subsample:
      self._episodes.append(_nested_stack(self._ep_buffer))
      return True
    else:
      return False

  def collect_n_episodes(self, n):
    for _ in tqdm(range(n)):
      found = False
      while not found:
        found = self.collect_episode()

  def _record_step(self, timestep, action):
    reward = np.array(timestep.reward or 0, np.float32)
    discount = tf.constant(timestep.discount if timestep.discount is not None else 1, tf.float32)
    self._ep_buffer.append((self._prev_observation,
                            action,
                            reward,
                            discount))

  def make_tf_dataset(self):
    self.types = tree.map_structure(lambda x: x.dtype, self._episodes[0])
    # the shapes are given by None since the ep length varies
    self.shapes = ((None, self._episodes[0][0].shape[1]), (None,), (None,), (None,))

    self.ds = tf.data.Dataset.from_generator(lambda: self._episodes, self.types, self.shapes)
    return self.ds

  @tf.autograph.experimental.do_not_convert
  def save(self, directory='datasets', overwrite=False):
    if not overwrite:
      directory = os.path.join(directory, str(int(time.time())))
    os.makedirs(directory, exist_ok=True)

    spec = {'types': self.types,
            'shapes': self.shapes}

    spec_file = os.path.join(directory, 'spec.pkl')
    with open(spec_file, 'wb') as f:
      pickle.dump(spec, f)

    for i, _ in enumerate(self.ds.element_spec):
      file_path = os.path.join(directory, f'offline_data.{i}.tfrecord')
      ds_i = self.ds.map(lambda *args: args[i]).map(tf.io.serialize_tensor)
      writer = tf.data.experimental.TFRecordWriter(file_path, compression_type='GZIP')
      writer.write(ds_i)


def compute_empirical_policy(dataset: tf.data.Dataset):
  """
  Input dataset and this function will return a tensorflow lookup table
  that maps observation to a tensor of action dimension that contains
  probabilities of each actions being taken for each observation within
  the given dataset.
  """
  print('_____Evaluating counts for all state action pairs_____ ')
  empirical_policy = {}
  dataset_len = sum([1 for _ in dataset])
  for e in tqdm(dataset, total=dataset_len):
    for o, a in zip(e[0], e[1]):
      if empirical_policy.get(str(o)) is not None:
        empirical_policy[str(o)][0][a] += 1
      else:
        empirical_policy[str(o)] = (np.zeros(3), o)

  counts, obs = zip(*empirical_policy.values())  # unzip

  def _normalize_to_tensor(array):
    total = sum(array)
    normed = array / total if total else array
    return tf.convert_to_tensor(normed, dtype=tf.float32)

  counts = [_normalize_to_tensor(i) for i in counts]
  obs = [str(o) for o in obs]

  table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.string,
                                                value_dtype=tf.float32,
                                                default_value=[-1., -1., -1.],
                                                empty_key='',
                                                deleted_key='del')
  table.insert(obs, counts)
  return table


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
  return tf.data.Dataset.zip(tuple(parts))


def preprocess_dataset(dataset: tf.data.Dataset, batch_size: int, n_step_returns: int, discount: float):
  d_len = sum([1 for _ in dataset])
  dataset = dataset.map(lambda *x:
                               n_step_transition_from_episode(*x, n_step=n_step_returns,
                                                              additional_discount=discount))
  dataset = dataset.repeat().shuffle(d_len).batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def load_wb_model(model_name: str, model_tag: str, dir: str = 'network', wandb_project_path: str = WANDB_PROJECT_PATH):
  """
  Load a model from wandb artifact by providing the model name and tag.
  Args:
      model_name: name used to save the model - default is `acme_checkpoint`
      model_tag: this is the version of the model to be loaded in the form of v1, v2, etc...
      dir: local directory where the checkpoint should be downloaded
      wandb_project_path: path to the wandb project used

  Returns:
      loaded_network: tf model that can be used for inference
  """
  wb_run = wandb.init()
  wb_path = wandb_project_path.format(model_name, model_tag)
  logging.info("Downloading model artifact from: " + wb_path)
  artifact = wb_run.use_artifact(wb_path, type='model')
  download_dir = artifact.download()
  logging.info("Model checkpoint downloaded to: {}".format(download_dir))
  model = os.path.join(download_dir, f'snapshots/{dir}')
  loaded_network = tf.saved_model.load(model)
  return loaded_network


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
