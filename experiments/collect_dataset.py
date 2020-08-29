import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import wandb
from absl import app, flags,logging

import trfl
import tensorflow as tf
import sonnet as snt
from acme import specs
from acme.tf import networks
from acme.agents.tf import actors

from utils import RandomActor, DemonstrationRecorder

from acme.wrappers import gym_wrapper
from gym_minigrid.wrappers import FullyObsWrapper
from custom_env_wrappers import ImgFlatObsWrapper, CustomSinglePrecisionWrapper


flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('save_dir', 'datasets/random', 'Direcotry to which the dataset will be saved.')
flags.DEFINE_integer('n_episodes', 100, 'Number of episodes to collect.')
flags.DEFINE_integer('n_episode_steps', 500, 'Max number of steps in each episode.')
flags.DEFINE_string('model_name', '', 'Name of the artifact with the model.')
flags.DEFINE_string('model_tag', 'latest', 'Specific version of the model.')
flags.DEFINE_integer('epsilon', 0.1, 'Epsilon for the e-greedy behavioural policy')
flags.DEFINE_boolean('stochastic', False, 'Sample the policy if it is parametrized by tfd.')
FLAGS = flags.FLAGS

WANDB_PROJECT_PATH = 'kmeco/offline-rl/{}:{}'


def main(_):
    # Create an environment and create the spec.
    raw_env = gym.make(FLAGS.environment_name)
    raw_env.action_space.n = 3
    raw_env.max_steps = FLAGS.n_episode_steps
    environment = ImgFlatObsWrapper(FullyObsWrapper(raw_env))
    environment = gym_wrapper.GymWrapper(environment)
    environment = CustomSinglePrecisionWrapper(environment)
    environment_spec = specs.make_environment_spec(environment)

    if FLAGS.model_name:
        wb_run = wandb.init()
        wb_path = WANDB_PROJECT_PATH.format(FLAGS.model_name, FLAGS.model_tag)
        logging.info("Downloading model artifact from: " + wb_path)
        artifact = wb_run.use_artifact(wb_path, type='model')
        download_dir = artifact.download()
        logging.info("Model checkpoint downloaded to: {}".format(download_dir))
        model = os.path.join(download_dir, 'snapshots/network')
        loaded_network = tf.saved_model.load(model)
        if FLAGS.stochastic:
            head = networks.StochasticSamplingHead()
        else:
            head = lambda q: trfl.epsilon_greedy(q, epsilon=FLAGS.epsilon).sample()

        policy_network = snt.Sequential([
            loaded_network,
            head,
        ])
        actor = actors.FeedForwardActor(policy_network)

    else:
        actor = RandomActor(environment_spec)

    recorder = DemonstrationRecorder(environment, actor)

    recorder.collect_n_episodes(FLAGS.n_episodes)
    recorder.make_tf_dataset()
    recorder.save(FLAGS.save_dir)


if __name__ == '__main__':
  app.run(main)
