#@title Import modules.
#python3
import copy
import time
import os

import wandb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl import app
from absl import flags
from tqdm import tqdm

from acme.agents.tf import actors

from acme import EnvironmentLoop
from acme.utils import counting
from acme import specs

import trfl
import tensorflow as tf
import sonnet as snt
from utils import n_step_transition_from_episode, load_tf_dataset, _build_environment, _build_custom_loggers

from acme.tf import utils as tf2_utils
from cql.learning import CQLLearner

# general run config
flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('logs_dir', 'logs-CQL-0', 'TB logs directory')
flags.DEFINE_string('logs_tag', 'tag', 'Tag a specific run for logging in TB.')
flags.DEFINE_boolean('wandb', False, 'Whether to log results to wandb.')
flags.DEFINE_string('dataset_dir', 'datasets', 'Directory containing an offline dataset.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
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
flags.DEFINE_float('cql_alpha', 1.0, 'Scaling parameter for the offline loss regularizer.')
flags.DEFINE_string('policy_improvement_mode', 'binary', 'Defines how the advantage is processed.')
FLAGS = flags.FLAGS
config = FLAGS.flag_values_dict()


def main(_):
    for _ in range(FLAGS.n_random_runs):
        wb_run = wandb.init(project="offline-rl",
                            group=FLAGS.logs_tag,
                            id=str(int(time.time())),
                            config=config,
                            reinit=FLAGS.acme_id is None) if FLAGS.wandb else None

        if FLAGS.seed:
            tf.random.set_seed(FLAGS.seed)
        # Create an environment and grab the spec.
        environment = _build_environment(FLAGS.environment_name)
        environment_spec = specs.make_environment_spec(environment)

        # Load demonstration dataset.
        dataset, empirical_policy = load_tf_dataset(directory=FLAGS.dataset_dir)
        dataset = dataset.map(lambda *x:
                              n_step_transition_from_episode(*x, n_step=FLAGS.n_step_returns,
                                                             additional_discount=FLAGS.discount))
        dataset = dataset.repeat().batch(FLAGS.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        network = snt.Sequential([
          snt.Flatten(),
          snt.nets.MLP([128, 64, 32, environment_spec.actions.num_values])
        ])
        # Create a target network.
        target_network = copy.deepcopy(network)

        policy_network = snt.Sequential([
            network,
            lambda q: trfl.epsilon_greedy(q, epsilon=FLAGS.epsilon).sample(),
        ])

        counter = counting.Counter()
        learner_counter = counting.Counter(counter)

        # Create the actor which defines how we take actions.
        evaluation_network = actors.FeedForwardActor(policy_network)

        # Ensure that we create the variables before proceeding (maybe not needed).
        tf2_utils.create_variables(network, [environment_spec.observations])
        tf2_utils.create_variables(target_network, [environment_spec.observations])

        disp, disp_loop = _build_custom_loggers(wb_run, FLAGS.logs_tag)

        eval_loop = EnvironmentLoop(
            environment=environment,
            actor=evaluation_network,
            counter=counter,
            logger=disp_loop)

        learner = CQLLearner(
            network=network,
            target_network=target_network,
            discount=0.99,
            importance_sampling_exponent=0.2,
            learning_rate=FLAGS.learning_rate,
            cql_alpha=FLAGS.cql_alpha,
            target_update_period=100,
            empirical_policy=empirical_policy,
            dataset=dataset,
            logger=disp,
            counter=learner_counter,
            checkpoint_subpath=os.path.join(wandb.run.dir, "acme/")
        )

        # Run the environment loop.
        for _ in tqdm(range(FLAGS.epochs)):
            for _ in range(FLAGS.evaluate_every):
                learner.step()
            eval_loop.run(FLAGS.evaluation_episodes)
        learner.save()


if __name__ == '__main__':
    app.run(main)

