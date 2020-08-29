import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import app, flags

import trfl
import sonnet as snt
from acme import specs
from acme.tf import networks
from acme.agents.tf import actors

from utils import RandomActor, DemonstrationRecorder, _build_environment, load_wb_model


flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('save_dir', 'datasets/random', 'Direcotry to which the dataset will be saved.')
flags.DEFINE_integer('n_episodes', 100, 'Number of episodes to collect.')
flags.DEFINE_integer('max_steps_per_episode', 500, 'Max number of steps in each episode.')
flags.DEFINE_string('model_name', '', 'Name of the artifact with the model.')
flags.DEFINE_string('model_tag', 'latest', 'Specific version of the model.')
flags.DEFINE_integer('epsilon', 0.1, 'Epsilon for the e-greedy behavioural policy')
flags.DEFINE_boolean('stochastic', False, 'Sample the policy if it is parametrized by tfd.')
FLAGS = flags.FLAGS


def main(_):
    # Create an environment and create the spec.
    environment, environment_spec = _build_environment(FLAGS.environment_name,
                                                       max_steps=FLAGS.max_steps_per_episode)

    if FLAGS.model_name:
        loaded_network = load_wb_model(FLAGS.model_name, FLAGS.model_tag)

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
