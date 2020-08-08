#@title Import modules.
#python3
import os
import time

import gym
from gym_minigrid.wrappers import FullyObsWrapper
from custom_env_wrappers import ImgFlatObsWrapper, CustomSinglePrecisionWrapper

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
flags.DEFINE_string('logs_dir', 'logs-CQL-0', 'TB logs directory')
flags.DEFINE_string('logs_tag', 'tag', 'Tag a specific run for logging in TB.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')
flags.DEFINE_float('epsilon', 0.3, 'Epsilon for e-greedy actor policy.')
flags.DEFINE_float('cql_alpha', 1e-3, 'Scaling parameter for the offline loss regularizer.')
flags.DEFINE_integer('n_episodes', 1000, 'Number of episodes to train for.')
flags.DEFINE_integer('n_steps', 1, 'Number of steps to bootstrap on when calculating TD(n)')

FLAGS = flags.FLAGS


#util
def _build_custom_loggers():
    logs_dir = os.path.join(FLAGS.logs_dir, str(int(time.time())) + "_" + FLAGS.logs_tag)
    terminal_logger = loggers.TerminalLogger(label='learner', time_delta=10)
    tb_logger = TFSummaryLogger(logdir=logs_dir, label='learner')
    disp = loggers.Dispatcher([terminal_logger, tb_logger])

    terminal_logger = loggers.TerminalLogger(label='Loop', time_delta=10)
    tb_logger = TFSummaryLogger(logdir=logs_dir, label='Loop')
    disp_loop = loggers.Dispatcher([terminal_logger, tb_logger])

    return disp, disp_loop


def _build_environment(n_actions=3, max_steps=500):
    raw_env = gym.make(FLAGS.environment_name)
    raw_env.action_space.n = n_actions
    raw_env.max_steps = max_steps
    env = ImgFlatObsWrapper(FullyObsWrapper(raw_env))
    env = gym_wrapper.GymWrapper(env)
    env = CustomSinglePrecisionWrapper(env)
    return env


def main(_):
  # Create an environment and grab the spec.
  environment = _build_environment()
  environment_spec = specs.make_environment_spec(environment)

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, environment_spec.actions.num_values])
  ])

  disp, disp_loop = _build_custom_loggers()

  # Construct the agent.
  agent = CQL(
      environment_spec=environment_spec,
      network=network,
      n_step=FLAGS.n_steps,
      epsilon=FLAGS.epsilon,
      cql_alpha=FLAGS.cql_alpha,
      logger=disp)

  # Run the environment loop.
  loop = EnvironmentLoop(environment, agent, logger=disp_loop)
  loop.run(num_episodes=FLAGS.n_episodes)  # pytype: disable=attribute-error
  agent.save()

if __name__ == '__main__':
  app.run(main)

