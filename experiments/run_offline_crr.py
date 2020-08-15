#@title Import modules.
#python3
import copy
import os
import time

from absl import app
from absl import flags
from tqdm import tqdm

import gym
from acme.agents.tf import actors
from gym_minigrid.wrappers import FullyObsWrapper
from custom_env_wrappers import ImgFlatObsWrapper, CustomSinglePrecisionWrapper

from acme.wrappers import gym_wrapper
from acme import EnvironmentLoop
from acme.utils import loggers, counting
from acme import specs
from acme.utils.loggers.tf_summary import TFSummaryLogger

import tensorflow as tf
import sonnet as snt
import tensorflow_probability as tfp
from utils import n_step_transition_from_episode, load_tf_dataset

from acme.tf import utils as tf2_utils, networks
from crr.learning import CRRLearner


flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('logs_dir', 'logs-CQL-0', 'TB logs directory')
flags.DEFINE_string('logs_tag', 'tag', 'Tag a specific run for logging in TB.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')
flags.DEFINE_string('dataset_dir', 'datasets', 'Directory containing an offline dataset.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('epsilon', 0.3, 'Epsilon for the epsilon greedy in the env.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('cql_alpha', 1e-3, 'Scaling parameter for the offline loss regularizer.')
flags.DEFINE_integer('n_step_returns', 5, 'Bootstrap after n steps.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to run (samples only 1 transition per episode in each epoch).')
flags.DEFINE_string('policy_improvement_mode', 'binary', 'Defines how the advantage is processed.')

FLAGS = flags.FLAGS


#utils
def _build_custom_loggers():
    tag = str(int(time.time())) + \
            "_" + str(FLAGS.cql_alpha) + "|" \
            + str(FLAGS.n_step_returns) + "|" \
            + FLAGS.logs_tag

    logs_dir = os.path.join(FLAGS.logs_dir, tag)
    terminal_logger = loggers.TerminalLogger(label='Learner', time_delta=10)
    tb_logger = TFSummaryLogger(logdir=logs_dir, label='Learner')
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

    # Load demonstration dataset.
    dataset, empirical_policy = load_tf_dataset(directory=FLAGS.dataset_dir)
    dataset = dataset.map(lambda *x:
                          n_step_transition_from_episode(*x, n_step=FLAGS.n_step_returns,
                                                         additional_discount=1.))
    dataset = dataset.repeat().batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


    # Create the critic network.
    critic_network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([128, 64, 32, environment_spec.actions.num_values]),
    ])

    policy_network = snt.Sequential([
      copy.deepcopy(critic_network),
      tfp.distributions.Categorical
    ])

    behaviour_network = snt.Sequential([
      policy_network,
      networks.StochasticSamplingHead()
    ])

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    # Create the actor which defines how we take actions.
    evaluation_actor = actors.FeedForwardActor(behaviour_network)

    # Ensure that we create the variables before proceeding (maybe not needed).
    tf2_utils.create_variables(policy_network, [environment_spec.observations])
    tf2_utils.create_variables(critic_network, [environment_spec.observations])

    disp, disp_loop = _build_custom_loggers()

    eval_loop = EnvironmentLoop(
        environment=environment,
        actor=evaluation_actor,
        counter=counter,
        logger=disp_loop)

    learner = CRRLearner(
        policy_network=policy_network,
        critic_network=critic_network,
        dataset=dataset,
        policy_improvement_modes=FLAGS.policy_improvement_mode,
        logger=disp
    )

    # Run the environment loop.
    for _ in tqdm(range(FLAGS.epochs)):
        for _ in range(FLAGS.evaluate_every):
            learner.step()
        learner_counter.increment(learner_steps=FLAGS.evaluate_every)
        eval_loop.run(FLAGS.evaluation_episodes)
    learner.save()


if __name__ == '__main__':
    app.run(main)

