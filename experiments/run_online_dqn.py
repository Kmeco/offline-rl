#python3
import os
import uuid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

from absl import app
from absl import flags
import wandb

from acme import specs

from acme.agents.tf import dqn
from acme import EnvironmentLoop
import sonnet as snt

# Bsuite flags
from utils import _build_environment, _build_custom_loggers

flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_string('wandb_id', '', 'Specific wandb id if you wish to continue in a checkpoint.')
flags.DEFINE_string('logs_tag', 'tag', 'Tag a specific run for logging in TB.')
flags.DEFINE_boolean('wandb', True, 'Whether to log results to wandb.')
flags.DEFINE_float('epsilon', 0.3, 'Epsilon for e-greedy actor policy.')
flags.DEFINE_integer('n_episodes', 1000, 'Number of episodes to train for.')
flags.DEFINE_integer('n_step_returns', 1, 'Number of steps to bootstrap on when calculating TD(n)')
flags.DEFINE_integer('batch_size', 256, 'Batch size for the learner.')
flags.DEFINE_integer('ep_max_len', 500, 'Maximum length of an episode.')
FLAGS = flags.FLAGS


def main(_):
  wb_run = wandb.init(project="offline-rl",
                      group=FLAGS.logs_tag,
                      id=FLAGS.wandb_id or str(int(time.time())),
                      config=FLAGS.flag_values_dict(),
                      reinit=FLAGS.acme_id is None) if FLAGS.wandb else None

  # Create an environment and grab the spec.
  environment, environment_spec = _build_environment(FLAGS.environment_name, max_steps=FLAGS.ep_max_len)

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([128, 64, 32, environment_spec.actions.num_values])
  ])

  disp, disp_loop = _build_custom_loggers(wb_run)

  # Construct the agent.
  agent = dqn.DQN(
      environment_spec=environment_spec,
      network=network,
      batch_size=FLAGS.batch_size,
      n_step=FLAGS.n_step_returns,
      epsilon=FLAGS.epsilon,
      logger=disp
  )

  # Run the environment loop.
  loop = EnvironmentLoop(environment, agent, logger=disp_loop)
  loop.run(num_episodes=FLAGS.n_episodes)  # pytype: disable=attribute-error
  agent._checkpointer.save(force=True)
  wandb.save(agent._checkpointer._checkpoint_dir)
  wandb.run.summary.update({"checkpoint_dir": agent._checkpointer._checkpoint_dir})


if __name__ == '__main__':
  app.run(main)
