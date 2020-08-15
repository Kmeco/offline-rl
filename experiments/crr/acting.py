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

"""Generic actor implementation, using TensorFlow and Sonnet."""

from acme import adders
from acme import core
from acme import types
# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree

tfd = tfp.distributions


class CRRActor(core.Actor):
  """A feed-forward actor.

  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

  def __init__(
      self,
      policy_network: snt.Module
  ):
    """Initializes the actor.

    Args:
      policy_network: the policy to run.
    """

    self._policy_network = tf.function(policy_network)

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)

    # Forward the policy network.
    logits = self._policy_network(batched_obs)
    a_dist = tfd.Categorical(logits=logits)

    # If the policy network parameterises a distribution, sample from it.
    def maybe_sample(output):
      if isinstance(output, tfd.Distribution):
        output = output.sample()
      return output

    policy_output = tree.map_structure(maybe_sample, policy_output)

    # Convert to numpy and squeeze out the batch dimension.
    action = tf2_utils.to_numpy_squeeze(policy_output)

    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    if self._adder:
      self._adder.add(action, next_timestep)

  def update(self):
    if self._variable_client:
      self._variable_client.update()
