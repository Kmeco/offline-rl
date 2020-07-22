"""Environment wrapper which converts double-to-single precision
for compatibility with reverb. """

from acme import specs
from acme import types
from acme.wrappers import base

from gym import spaces
import dm_env
import numpy as np
import tree
import gym


class SinglePrecisionWrapper(base.EnvironmentWrapper):
  """Wrapper which converts environments from double- to single-precision."""

  def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    return timestep._replace(
        reward=_convert_value(timestep.reward),
        discount=_convert_value(timestep.discount),
        observation=_convert_value(timestep.observation).T.flatten())

  def step(self, action) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.step(action))

  def reset(self) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.reset())

  def action_spec(self):
    return _convert_spec(self._environment.action_spec())

  def discount_spec(self):
    return _convert_spec(self._environment.discount_spec())

  def observation_spec(self):
    return _convert_spec(self._environment.observation_spec())

  def reward_spec(self):
    return _convert_spec(self._environment.reward_spec())


def _convert_spec(nested_spec: types.NestedSpec) -> types.NestedSpec:
  """Convert a nested spec."""

  def _convert_single_spec(spec: specs.Array):
    """Convert a single spec."""
    if np.issubdtype(spec.dtype, np.float64):
      dtype = np.float32
    elif np.issubdtype(spec.dtype, np.int64):
      dtype = np.int32
    else:
      dtype = np.float32
    return spec.replace(dtype=dtype)

  return tree.map_structure(_convert_single_spec, nested_spec)


def _convert_value(nested_value: types.Nest) -> types.Nest:
  """Convert a nested value given a desired nested spec."""

  def _convert_single_value(value):
    if value is not None:
      value = np.array(value, copy=False)
      if np.issubdtype(value.dtype, np.float64):
        value = np.array(value, copy=False, dtype=np.float32)
      else:
        value = np.array(value, copy=False, dtype=np.float32)
    return value

  return tree.map_structure(_convert_single_value, nested_value)


class ImgFlatObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        size = np.prod(env.observation_space.spaces['image'].shape)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size, ),
            dtype=np.float32
        )

    def observation(self, obs):
        value = obs['image'].T.flatten()
        return np.array(value, copy=False, dtype=np.float32)