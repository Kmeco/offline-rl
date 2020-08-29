#python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import wandb

from absl import app, flags, logging

from acme import specs

from cql.agent import CQL
from acme.utils import counting
from acme import EnvironmentLoop
import sonnet as snt

from utils import _build_environment, _build_custom_loggers

flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_string('logs_dir', 'logs-CQL-0', 'TB logs directory')
flags.DEFINE_string('logs_tag', 'tag', 'Tag a specific run for logging in TB.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')
flags.DEFINE_float('epsilon', 0.3, 'Epsilon for e-greedy actor policy.')
flags.DEFINE_float('discount', 0.99, 'Discount rate for the learner.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_integer('max_replay_size', 10000, 'Maximum number of trajectories kept in the replay buffer.')
flags.DEFINE_integer('samples_per_insert', 32, 'How many updates to do for each env step.')
flags.DEFINE_float('cql_alpha', 1e-3, 'Scaling parameter for the offline loss regularizer.')
flags.DEFINE_integer('n_episodes', 1000, 'Number of episodes to train for.')
flags.DEFINE_integer('n_steps', 1, 'Number of steps to bootstrap on when calculating TD(n)')
flags.DEFINE_boolean('wandb', True, 'Whether to log results to wandb.')
flags.DEFINE_string('wandb_id', '', 'Specific wandb id if you wish to continue in a checkpoint.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for the learner.')
flags.DEFINE_integer('ep_max_len', 500, 'Maximum length of an episode.')
FLAGS = flags.FLAGS

WANDB_PROJECT_PATH = 'kmeco/offline-rl/{}:latest'


def init_or_resume():
  wb_run = wandb.init(project="offline-rl",
                      group=FLAGS.logs_tag,
                      id=FLAGS.wandb_id or str(int(time.time())),
                      config=FLAGS.flag_values_dict(),
                      resume=FLAGS.wandb_id is not None,
                      reinit=True) if FLAGS.wandb else None
  if FLAGS.wandb_id:
    checkpoint_dir = wandb.run.summary['checkpoint_dir']
    group = wandb.run.summary['group']

    logging.info("Downloading model artifact from: " + WANDB_PROJECT_PATH.format(group))
    artifact = wb_run.use_artifact(WANDB_PROJECT_PATH.format(group), type='model')
    download_dir = artifact.download(root=checkpoint_dir)
    FLAGS.acme_id = checkpoint_dir.split('/')[-2]
    logging.info("Model checkpoint downloaded to: {}".format(download_dir))
  return wb_run


def main(_):
  wb_run = init_or_resume()

  # Create an environment and grab the spec.
  environment, environment_spec = _build_environment(FLAGS.environment_name, max_steps = FLAGS.ep_max_len)

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([128, 64, 32, environment_spec.actions.num_values]) # TODO: try sigmoid
  ])

  disp, disp_loop = _build_custom_loggers(wb_run)

  counter = counting.Counter()

  # Construct the agent.
  agent = CQL(
      environment_spec=environment_spec,
      network=network,
      n_step=FLAGS.n_steps,
      epsilon=FLAGS.epsilon,
      discount=FLAGS.discount,
      cql_alpha=FLAGS.cql_alpha,
      max_replay_size=FLAGS.max_replay_size,
      samples_per_insert=FLAGS.samples_per_insert,
      learning_rate=FLAGS.learning_rate,
      counter=counter,
      logger=disp)

  # Run the environment loop.
  loop = EnvironmentLoop(environment, agent, counter=counter, logger=disp_loop)
  loop.run(num_episodes=FLAGS.n_episodes)  # pytype: disable=attribute-error
  agent.save(tag=FLAGS.logs_tag)


if __name__ == '__main__':
  app.run(main)
