#@title Import modules.
#python3
import pyvirtualdisplay

import gym
from gym_minigrid.wrappers import FullyObsWrapper
from custom_env_wrappers import ImgFlatObsWrapper, SinglePrecisionWrapper

# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()


from absl import app
from absl import flags

from acme.wrappers import gym_wrapper
from acme.utils import loggers
from acme import specs
from acme.utils.loggers.tf_summary import TFSummaryLogger

from cql.agent import CQL
from acme import EnvironmentLoop
import sonnet as snt

# Bsuite flags
flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')
FLAGS = flags.FLAGS


def main(_):
  # Create an environment and grab the spec.
  raw_env = gym.make(FLAGS.environment_name)
  raw_env.action_space.n = 3
  raw_env.max_steps = 500
  environment = ImgFlatObsWrapper(FullyObsWrapper(raw_env))
  environment = gym_wrapper.GymWrapper(environment)
  environment = SinglePrecisionWrapper(environment)
  environment_spec = specs.make_environment_spec(environment)

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, environment_spec.actions.num_values])
  ])

  # Construct the agent.
  agent = CQL(
      environment_spec=environment_spec,
      network=network,
      epsilon=0.3,
      logger=loggers.TerminalLogger(label='Learner', time_delta=10.))

  terminal_logger = loggers.TerminalLogger(label='terminal', time_delta=10)
  tb_logger = TFSummaryLogger(logdir='logsV2', label='summary')
  disp = loggers.Dispatcher([terminal_logger, tb_logger])

  # Run the environment loop.
  loop = EnvironmentLoop(environment, agent, logger=disp)
  loop.run(num_episodes=1000)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)

