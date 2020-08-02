#@title Import modules.
#python3
import copy

from absl import app
from absl import flags
from tqdm import tqdm

import gym
from acme.agents.tf import actors
from gym_minigrid.wrappers import FullyObsWrapper
from custom_env_wrappers import ImgFlatObsWrapper, SinglePrecisionWrapper

from acme.wrappers import gym_wrapper
from acme import EnvironmentLoop
from acme.utils import loggers, counting
from acme import specs
from acme.utils.loggers.tf_summary import TFSummaryLogger

import trfl
import tensorflow as tf
import sonnet as snt
from utils import n_step_transition_from_episode, load_tf_dataset

from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from cql.learning import CQLLearner


flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('logs_dir', 'logs-CQL-0', 'TB logs directory')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')
flags.DEFINE_string('dataset_dir', 'datasets', 'Directory containing an offline dataset.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('epsilon', 0.3, 'Epsilon for the epsilon greedy in the env.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_integer('n_step_returns', 5, 'Bootstrap after n steps.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_string('epochs', 100, 'Number of epochs to run (samples only 1 transition per episode in each epoch).')
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

    # Load demonstration dataset.
    dataset, empirical_policy = load_tf_dataset(directory=FLAGS.dataset_dir)
    dataset = dataset.map(lambda *x:
                          n_step_transition_from_episode(*x, n_step=FLAGS.n_step_returns,
                                                         additional_discount=1.))
    dataset = dataset.repeat().batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, environment_spec.actions.num_values])
    ])
    # Create a target network.
    target_network = copy.deepcopy(network)

    policy_network = snt.Sequential([
        network,
        lambda q: trfl.epsilon_greedy(q, epsilon=FLAGS.epsilon).sample(),
    ])

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    # Create the actor which defines how we take actions.
    evaluation_network = actors.FeedForwardActor(policy_network)

    # Ensure that we create the variables before proceeding (maybe not needed).
    tf2_utils.create_variables(network, [environment_spec.observations])
    tf2_utils.create_variables(target_network, [environment_spec.observations])

    terminal_logger = loggers.TerminalLogger(label='evaluation', time_delta=10)
    tb_logger = TFSummaryLogger(logdir=FLAGS.logs_dir, label='evaluation')
    disp_loop = loggers.Dispatcher([terminal_logger, tb_logger])

    eval_loop = EnvironmentLoop(
        environment=environment,
        actor=evaluation_network,
        counter=counter,
        logger=disp_loop)

    terminal_logger = loggers.TerminalLogger(label='learner', time_delta=10)
    tb_logger = TFSummaryLogger(logdir=FLAGS.logs_dir, label='learner')
    disp = loggers.Dispatcher([terminal_logger, tb_logger])

    learner = CQLLearner(
        network=network,
        target_network=target_network,
        discount=0.99,
        importance_sampling_exponent=0.2,
        learning_rate=1e-3,
        cql_alpha=0.01,
        target_update_period=100,
        empirical_policy=empirical_policy,
        dataset=dataset,
        logger=disp
    )

    # Run the environment loop.
    for _ in tqdm(range(FLAGS.epochs)):
        for _ in range(FLAGS.evaluate_every):
            learner.step()
        learner_counter.increment(learner_steps=FLAGS.evaluate_every)
        eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)

