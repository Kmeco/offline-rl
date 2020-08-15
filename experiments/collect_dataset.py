import gym
from absl import app
from absl import flags
from acme.wrappers import gym_wrapper
from utils import RandomActor, DemonstrationRecorder
from acme import specs


from gym_minigrid.wrappers import FullyObsWrapper
from custom_env_wrappers import ImgFlatObsWrapper, CustomSinglePrecisionWrapper


flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('save_dir', 'datasets/random', 'Direcotry to which the dataset will be saved.')
flags.DEFINE_integer('n_episodes', 100, 'Number of episodes to collect.')
flags.DEFINE_integer('n_episode_steps', 500, 'Max number of steps in each episode.')
FLAGS = flags.FLAGS


def main(_):
    # Create an environment and create the spec.
    raw_env = gym.make(FLAGS.environment_name)
    raw_env.action_space.n = 3
    raw_env.max_steps = FLAGS.n_episode_steps
    environment = ImgFlatObsWrapper(FullyObsWrapper(raw_env))
    environment = gym_wrapper.GymWrapper(environment)
    environment = CustomSinglePrecisionWrapper(environment)
    environment_spec = specs.make_environment_spec(environment)

    actor = RandomActor(environment_spec)
    recorder = DemonstrationRecorder(environment, actor)

    recorder.collect_n_episodes(FLAGS.n_episodes)
    recorder.make_tf_dataset()
    recorder.save(FLAGS.save_dir)


if __name__ == '__main__':
  app.run(main)