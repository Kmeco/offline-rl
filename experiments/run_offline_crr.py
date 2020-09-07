#python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import time

from absl import app, flags, logging
from tqdm import tqdm
import wandb

from acme.agents.tf import actors

from acme.environment_loop import EnvironmentLoop
from acme.utils import counting
from acme.tf import utils as tf2_utils
from acme.tf.networks import StochasticSamplingHead

import tensorflow as tf
import sonnet as snt
import tensorflow_probability as tfp
import networks
from utils import load_tf_dataset, _build_environment, _build_custom_loggers, \
    preprocess_dataset, compute_empirical_policy
from visualization import evaluate_q, visualize_policy

from crr.learning import CRRLearner


# general run config
flags.DEFINE_string('environment_name', 'MiniGrid-Empty-6x6-v0', 'MiniGrid env name.')
flags.DEFINE_string('logs_tag', 'tag', 'Tag a specific run for logging in TB.')
flags.DEFINE_boolean('wandb', True, 'Whether to log results to wandb.')
flags.DEFINE_string('wandb_id', '', 'Specific wandb id if you wish to continue in a checkpoint.')
flags.DEFINE_string('dataset_dir', 'datasets', 'Directory containing an offline dataset.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_integer('max_eval_episode_len', 100, 'Evaluation episodes.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to run (samples only 1 transition per episode in each epoch).')
flags.DEFINE_integer('seed', 1234, 'Random seed for replicable results. Set to 0 for no seed.')

# general learner config
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_boolean('greedy', False, 'Should act greedily or sample the policy during online evaluation.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount factor.')
flags.DEFINE_integer('n_step_returns', 1, 'Bootstrap after n steps.')

# specific config
flags.DEFINE_float('crr_beta', 1.0, 'Param for calculating the policy improvement coefficient.')
flags.DEFINE_float('cql_alpha', 0.0, 'Scaling parameter for the offline loss regularizer.')
flags.DEFINE_string('policy_improvement_mode', 'binary', 'Defines how the advantage is processed.')
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

    if FLAGS.seed:
        tf.random.set_seed(FLAGS.seed)

    # Create an environment and grab the spec.
    environment, env_spec = _build_environment(FLAGS.environment_name,
                                               max_steps=FLAGS.max_eval_episode_len)

    # Load demonstration dataset.
    raw_dataset = load_tf_dataset(directory=FLAGS.dataset_dir)
    empirical_policy = compute_empirical_policy(raw_dataset)

    dataset = preprocess_dataset(raw_dataset,
                                 FLAGS.batch_size,
                                 FLAGS.n_step_returns,
                                 FLAGS.discount)

    # Create the policy and critic networks.
    critic_network = networks.get_default_critic(env_spec)

    policy_network = snt.Sequential([
      copy.deepcopy(critic_network),
      tfp.distributions.Categorical
    ])

    if FLAGS.greedy:
      head = networks.GreedyHead()
    else:
      head = StochasticSamplingHead()

    behaviour_network = snt.Sequential([
      policy_network,
      head
    ])

    # Ensure that we create the variables before proceeding (maybe not needed).
    tf2_utils.create_variables(policy_network, [env_spec.observations])
    tf2_utils.create_variables(critic_network, [env_spec.observations])

    # Create the actor which defines how we take actions.
    evaluation_actor = actors.FeedForwardActor(behaviour_network)

    counter = counting.Counter()

    disp, disp_loop = _build_custom_loggers(wb_run)

    eval_loop = EnvironmentLoop(
        environment=environment,
        actor=evaluation_actor,
        counter=counter,
        logger=disp_loop)

    learner = CRRLearner(
        policy_network=policy_network,
        critic_network=critic_network,
        dataset=dataset,
        discount=0.99,
        policy_improvement_modes=FLAGS.policy_improvement_mode,
        beta=FLAGS.crr_beta,
        cql_alpha=FLAGS.cql_alpha,
        empirical_policy=empirical_policy,
        logger=disp,
        counter=counter
    )

    # Run the environment loop.
    for e in tqdm(range(FLAGS.epochs)):
        for _ in range(FLAGS.evaluate_every):
            learner.step()
        eval_loop.run(FLAGS.evaluation_episodes)
        # Visualization of the policy
        Q = evaluate_q(learner._critic_network, environment)
        plot = visualize_policy(Q, environment)
        wb_run.log({'chart': plot, 'epoch_counter': e})

    learner.save(tag=FLAGS.logs_tag)


if __name__ == '__main__':
    app.run(main)

