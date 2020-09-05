# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN learner implementation."""

import time
import copy
from typing import Dict, List, Optional

import acme
import wandb
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import trfl

tfd = tfp.distributions


class CRRLearner(acme.Learner, tf2_savers.TFSaveable):
  def __init__(
      self,
      policy_network: snt.Module,
      critic_network: snt.Module,
      dataset: tf.data.Dataset,
      discount: float,
      behavior_network: Optional[snt.Module] = None,
      cwp_network: Optional[snt.Module] = None,
      policy_optimizer: Optional[snt.Optimizer] = snt.optimizers.Adam(1e-4),
      critic_optimizer: Optional[snt.Optimizer] = snt.optimizers.Adam(1e-4),
      target_update_period: int = 100,
      policy_improvement_modes: str = 'exp',
      ratio_upper_bound: float = 20.,
      beta: float = 1.0,
      cql_alpha: float = 0.0,
      translate_lse: float = 100.,
      empirical_policy: dict = None,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      checkpoint_subpath: str = '~/acme/'
  ):
    """Initializes the learner.

    Args:
      network: the online Q network (the one being optimized)
      target_network: the target Q critic (which lags behind the online net).
      discount: discount to use for TD updates.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      learning_rate: learning rate for the q-network update.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      dataset: dataset to learn from, whether fixed or from a replay buffer (see
        `acme.datasets.reverb.make_dataset` documentation).
      huber_loss_parameter: Quadratic-linear boundary for Huber loss.
      replay_client: client to replay to allow for updating priorities.
      counter: Counter object for (potentially distributed) counting.
      logger: Logger object for writing logs to.
      checkpoint: boolean indicating whether to checkpoint the learner.
    """

    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    # Store online and target networks.
    self._policy_network = policy_network
    self._critic_network = critic_network
    # Create a target networks.
    self._target_policy_network = copy.deepcopy(policy_network)
    self._target_critic_network = copy.deepcopy(critic_network)
    self._critic_optimizer = critic_optimizer
    self._policy_optimizer = policy_optimizer

    # self._alpha = tf.constant(cql_alpha, dtype=tf.float32)
    # self._emp_policy = empirical_policy
    self._target_update_period = target_update_period

    # Internalise the hyperparameters.
    self._discount = discount
    self._target_update_period = target_update_period
    # crr specific
    assert policy_improvement_modes in ['exp', 'binary', 'all'], 'Policy imp. mode must be one of {exp, binary, all}'
    self._policy_improvement_modes = policy_improvement_modes
    self._beta = beta
    self._ratio_upper_bound = ratio_upper_bound
    # cql specific
    self._alpha = tf.constant(cql_alpha, dtype=tf.float32)
    self._tr = tf.constant(translate_lse, dtype=tf.float32)
    if cql_alpha:
      assert empirical_policy is not None, 'Empirical behavioural policy must be specified with non-zero cql_alpha.'
    self._emp_policy = empirical_policy

    # Learner state.
    # Expose the variables.
    self._variables = {
        'critic': self._target_critic_network.variables,
        'policy': self._target_policy_network.variables,
    }

    # Internalise logging/counting objects.
    self._counter = counter or counting.Counter()
    self._counter.increment(learner_steps=0)
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Create a checkpointer and snapshoter object.
    self._checkpointer = tf2_savers.Checkpointer(
      objects_to_save=self.state,
      time_delta_minutes=10.,
      directory=checkpoint_subpath,
      subdirectory='crr_learner'
    )

    objects_to_save = {
      'raw_policy': policy_network,
      'critic': critic_network,
    }
    self._snapshotter = tf2_savers.Snapshotter(
      objects_to_save=objects_to_save, time_delta_minutes=10, directory=checkpoint_subpath)
    # Timestamp to keep track of the wall time.
    self._walltime_timestamp = time.time()

  # @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD."""

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data

    with tf.GradientTape(persistent=True) as tape:
      # Evaluate our networks.
      q_tm1 = self._critic_network(o_tm1)

      # The rewards and discounts have to have the same type as network values.
      r_t = tf.cast(r_t, q_tm1.dtype)
      r_t = tf.clip_by_value(r_t, -1., 1.)
      d_t = tf.cast(d_t, q_tm1.dtype) * tf.cast(self._discount, q_tm1.dtype)

      # ========================= Critic learning ============================
      target_a_t_dist = self._target_policy_network(o_t)
      target_a_t_probs = target_a_t_dist.probs_parameter()

      # Compute the target critic's Q-value of the sampled actions.
      q_t_target = self._target_critic_network(o_t)

      # Compute the loss
      critic_loss, extra = trfl.sarse(q_tm1, a_tm1, r_t, d_t, q_t_target, target_a_t_probs)

      # ========================= Actor learning =============================
      a_tm1_dist = self._policy_network(o_tm1)
      a_tm1_probs = a_tm1_dist.probs_parameter()
      expected_policy_q_tm1 = tf.reduce_sum(tf.multiply(q_tm1, a_tm1_probs), axis=1)

      policy_loss_batch = - tf.math.log(trfl.indexing_ops.batched_index(a_tm1_probs, a_tm1))
      qa_tm1 = trfl.indexing_ops.batched_index(q_tm1, a_tm1)

      advantage = qa_tm1 - expected_policy_q_tm1

      if self._policy_improvement_modes == 'exp':
        policy_loss_coef_t = tf.math.minimum(
          tf.math.exp(advantage / self._beta), self._ratio_upper_bound)
      elif self._policy_improvement_modes == 'binary':
        policy_loss_coef_t = tf.cast(advantage > 0, dtype=q_tm1.dtype)
      elif self._policy_improvement_modes == 'all':
        # Regress against all actions (effectively pure BC).
        policy_loss_coef_t = 1.

      policy_loss_coef_t = tf.stop_gradient(policy_loss_coef_t)
      policy_loss_batch *= policy_loss_coef_t

      policy_loss = tf.reduce_mean(policy_loss_batch)

      if self._alpha:
        policy_probs = self._emp_policy.lookup([str(o) for o in o_tm1])

        push_down = tf.reduce_logsumexp(q_tm1 * self._tr, axis=1) / self._tr  # soft-maximum of the q func
        push_up = tf.reduce_sum(policy_probs * q_tm1, axis=1)   # expected q value under behavioural policy

        critic_loss = critic_loss + self._alpha * (push_down - push_up)

      critic_loss = tf.reduce_mean(critic_loss, axis=0)

    # Compute gradients.
    critic_gradients = tape.gradient(critic_loss,
                                     self._critic_network.trainable_variables)
    policy_gradients = tape.gradient(policy_loss,
                                     self._policy_network.trainable_variables)

    # Delete the tape manually because of the persistent=True flag.
    del tape

    # Apply gradients.
    self._critic_optimizer.apply(critic_gradients,
                                 self._critic_network.trainable_variables)
    self._policy_optimizer.apply(policy_gradients,
                                 self._policy_network.trainable_variables)

    source_variables = (
        self._critic_network.variables + self._policy_network.variables)
    target_variables = (
        self._target_critic_network.variables +
        self._target_policy_network.variables)

    # Make online -> target network update ops.
    if tf.math.mod(self._counter.get_counts()['learner_steps'], self._target_update_period) == 0:
      for src, dest in zip(source_variables, target_variables):
        dest.assign(src)

    metrics = {
      'critic_loss': critic_loss,
      'policy_loss': policy_loss,
      'advantage': tf.reduce_mean(advantage),
      'q_variance': tf.reduce_mean(tf.math.reduce_variance(q_tm1, axis=1), axis=0),
      'q_average': tf.reduce_mean(q_tm1)
    }
    if self._alpha:
      metrics.update({'push_up': tf.reduce_mean(push_up, axis=0),
                      'push_down': tf.reduce_mean(push_down, axis=0)
                      })
    return metrics

  def step(self):
    # Run the learning step.
    fetches = self._step()

    # Update our counts and record it.
    new_timestamp = time.time()
    time_passed = new_timestamp - self._walltime_timestamp
    self._walltime_timestamp = new_timestamp

    # Update our counts and record it.
    counts = self._counter.increment(learner_steps=1, walltime=time_passed)
    fetches.update(counts)

    # Checkpoint and attempt to write the logs.
    self._checkpointer.save()
    self._snapshotter.save()

    self._logger.write(fetches)

  def save(self, tag='default'):
    self._snapshotter.save(force=True)
    self._checkpointer.save(force=True)

    artifact = wandb.Artifact(tag, type='model')
    dir_name = self._checkpointer._checkpoint_dir.split('checkpoints')[0]
    artifact.add_dir(dir_name)
    wandb.run.log_artifact(artifact)
    wandb.run.summary.update({"checkpoint_dir": dir_name, "group": tag})

  def get_variables(self, names: List[str]) -> List[np.ndarray]:
    return tf2_utils.to_numpy(self._variables)

  @property
  def state(self):
    """Returns the stateful parts of the learner for checkpointing."""
    return {
      'counter': self._counter,
      'policy': self._policy_network,
      'critic': self._critic_network,
      'target_policy': self._target_policy_network,
      'target_critic': self._target_critic_network,
      'policy_optimizer': self._policy_optimizer,
      'critic_optimizer': self._critic_optimizer,
    }


