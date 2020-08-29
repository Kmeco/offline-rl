#python3
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import trfl
from absl import app
from absl import flags
from tqdm import tqdm

from acme.agents.tf import actors

from acme.environment_loop import EnvironmentLoop
from acme.utils import counting
from acme import specs

import tensorflow as tf
import sonnet as snt
import tensorflow_probability as tfp
from utils import load_tf_dataset, _build_environment, _build_custom_loggers, \
  preprocess_dataset

from acme.tf import utils as tf2_utils
from acme.agents.tf.bc import learning
import wandb


# general run config
flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('logs_dir', 'logs-CQL-0', 'TB logs directory')
flags.DEFINE_string('logs_tag', 'tag', 'Tag a specific run for logging in TB.')
flags.DEFINE_boolean('wandb', False, 'Whether to log results to wandb.')
flags.DEFINE_string('wandb_id', '', 'Specific wandb id if you wish to continue in a checkpoint.')

flags.DEFINE_string('dataset_dir', 'datasets', 'Directory containing an offline dataset.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_integer('max_eval_episode_len', 100, 'Evaluation episodes.')

flags.DEFINE_integer('epochs', 100, 'Number of epochs to run (samples only 1 transition per episode in each epoch).')
flags.DEFINE_integer('seed', 1234, 'Random seed for replicable results. Set to 0 for no seed.')
flags.DEFINE_integer('n_random_runs', 1, 'Run n runs with different random seeds and track them under one wb group.')
# general learner config
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('epsilon', 0.3, 'Epsilon for the epsilon greedy in the env.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount factor.')
flags.DEFINE_integer('n_step_returns', 5, 'Bootstrap after n steps.')

# specific config
flags.DEFINE_float('crr_beta', 1.0, 'Param for calculating the policy improvement coefficient.')
flags.DEFINE_float('cql_alpha', 0.0, 'Scaling parameter for the offline loss regularizer.')
flags.DEFINE_string('policy_improvement_mode', 'binary', 'Defines how the advantage is processed.')
FLAGS = flags.FLAGS


def main(_):
    for n in range(FLAGS.n_random_runs):
        wb_run = wandb.init(project="offline-rl",
                            group=FLAGS.logs_tag,
                            id=str(int(time.time())),
                            config=FLAGS.flag_values_dict(),
                            reinit=FLAGS.acme_id is None) if FLAGS.wandb else None

        if FLAGS.seed:
            tf.random.set_seed(FLAGS.seed + n)

        # Create an environment and grab the spec.
        environment, environment_spec = _build_environment(FLAGS.environment_name)

        # Load demonstration dataset.
        dataset, empirical_policy = load_tf_dataset(directory=FLAGS.dataset_dir)
        dataset = preprocess_dataset(dataset, FLAGS.batch_size, FLAGS.n_step_returns, FLAGS.discount)

        # Create the policy and critic networks.
        policy_network = snt.Sequential([
          snt.Flatten(),
          snt.nets.MLP([128, 64, 32, environment_spec.actions.num_values]),
        ])

        # Ensure that we create the variables before proceeding (maybe not needed).
        tf2_utils.create_variables(policy_network, [environment_spec.observations])

        # If the agent is non-autoregressive use epsilon=0 which will be a greedy
        # policy.
        evaluator_network = snt.Sequential([
          policy_network,
          lambda q: trfl.epsilon_greedy(q, epsilon=FLAGS.epsilon).sample(),
        ])

        # Create the actor which defines how we take actions.
        evaluation_actor = actors.FeedForwardActor(evaluator_network)

        counter = counting.Counter()
        learner_counter = counting.Counter(counter)

        disp, disp_loop = _build_custom_loggers(wb_run, FLAGS.logs_tag)

        eval_loop = EnvironmentLoop(
            environment=environment,
            actor=evaluation_actor,
            counter=counter,
            logger=disp_loop)

        # The learner updates the parameters (and initializes them).
        learner = learning.BCLearner(
          network=policy_network,
          learning_rate=FLAGS.learning_rate,
          dataset=dataset,
          counter=learner_counter)

        # Run the environment loop.
        for _ in tqdm(range(FLAGS.epochs)):
            for _ in range(FLAGS.evaluate_every):
                learner.step()
            eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == '__main__':
    app.run(main)

