import base64
import copy

import IPython
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def _get_full_observation(state, base_obs):
  """"Given a 3dim compressed tuple state of the env,
  construct a full observation."""
  obs = copy.copy(base_obs)
  obs[0, state[0], state[1]] = 10
  obs[2, state[0], state[1]] = state[2]
  return tf.convert_to_tensor(obs.reshape(1, -1))


def _get_base_observation(environment):
  """Given an environment, return an observation with no agent."""
  obs = environment.reset().observation
  true_shape = _get_true_env_shape(environment)
  obs_base = obs.reshape(*true_shape)
  obs_base[0][obs_base[0] == 10] = 1.
  return obs_base, true_shape


def _get_true_env_shape(environment):
  """Beware this is a very custom made func."""
  # remove all wrappers
  raw_env = environment.environment.environment.env.env
  obs_space = raw_env.observation_space['image']
  shape_tuple = list(obs_space.shape)
  shape_tuple.reverse()
  return shape_tuple


def evaluate_q(critic_network, env):
  """
  Q is an array of size (d, n, n, a) where:
  d is the number of directions: 4
  n is the size of the grid: 6
  a is a number of actions: 3
  """
  obs_base, true_shape = _get_base_observation(env)

  Q = np.zeros((4, *true_shape[1:], true_shape[0]))
  for r, row in enumerate(obs_base[0]):
    for c, col in enumerate(row):
      if obs_base[0][r][c] == 1.:
        for k in range(4):
          state = (r, c, k)
          full_obs = _get_full_observation(state, obs_base)
          Q[k, r, c] = critic_network(full_obs).numpy()
  return Q


map_from_int_to_dir = lambda a: (r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$", r"$\uparrow$")[a]
map_from_action_to_name = lambda a, d: [r"$\circlearrowleft$", r"$\circlearrowright$", map_from_int_to_dir(d)][a]


def plot_values(values, colormap='pink', vmin=0.8, vmax=1):
  plt.imshow(values, interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax)
  plt.yticks([])
  plt.xticks([])
  plt.colorbar(ticks=[vmin, vmax])


def plot_state_value(q):
  vmin = np.quantile(q.flatten()[q.flatten() != 0], 0.05)
  vmax = np.max(q)
  v = 0.9 * np.max(q, axis=-1) + 0.1 * np.mean(q, axis=-1)
  plot_values(v, colormap='summer', vmin=vmin, vmax=vmax)
  plt.title("$v(s)$")


def plot_grid(obs):
  layout = (obs[0] == 2).astype(int) * (-1)
  plt.imshow(layout > -1, interpolation="nearest", cmap="YlOrRd_r")
  ax = plt.gca()
  ax.grid(0)
  plt.xticks([])
  plt.yticks([])
  goal = (obs[0] == 8.).astype(int)
  plt.text(
    np.argmax(np.max(goal, axis=0)), np.argmax(np.max(goal, axis=1)),
    r"$\mathbf{G}$", ha='center', va='center', size='large')
  h, w = layout.shape
  for y in range(h - 3):
    plt.plot([+0.5, w - 1.5], [y + 1.5, y + 1.5], '-k', lw=2, alpha=0.5)
  for x in range(w - 3):
    plt.plot([x + 1.5, x + 1.5], [+0.5, h - 1.5], '-k', lw=2, alpha=0.5)


def plot_greedy_policy(q, env, dir):
  greedy_actions = np.argmax(q, axis=2)
  obs, _ = _get_base_observation(env)
  plot_grid(obs)
  plt.title(f"The grid: {map_from_int_to_dir(dir)}", size='large')
  for r, row in enumerate(obs[0]):
    for c, col in enumerate(row):
      if obs[0][r][c] == 1.:
        action_name = map_from_action_to_name(greedy_actions[r, c], dir)
        plt.text(c, r, action_name, ha='center', va='center', size='large')


def visualize_policy(action_values, env):
  q = action_values
  fig = plt.figure(figsize=(17, 12))
  fig.subplots_adjust(wspace=0.3, hspace=0.3)
  vmin = np.quantile(q.flatten()[q.flatten() != 0], 0.05)
  vmax = np.max(action_values)
  dif = vmax - vmin
  for row in range(4):
    for a in range(3):
      plt.subplot(4, 5, (5 * (row) + a) + 1)
      plot_values(q[row, ..., a], vmin=vmin - 0.05 * dif, vmax=vmax + 0.05 * dif)
      action_name = map_from_action_to_name(a, row)
      plt.title(r"q(s," + action_name + r")")
    plt.subplot(4, 5, 5 * row + 4)
    plot_state_value(q[row])
    plt.subplot(4, 5, 5 * row + 5)
    plot_greedy_policy(q[row], env, row)
  return plt


def render(env):
  return env.environment.render(mode='rgb_array')


def display_video(frames, filename='temp.mp4'):
  """Save and display video."""
  # Write video
  with imageio.get_writer(filename, fps=60) as video:
    for frame in frames:
      video.append_data(frame)
  # Read video and display the video
  video = open(filename, 'rb').read()
  b64_video = base64.b64encode(video)
  video_tag = ('<video  width="320" height="240" controls alt="test" '
               'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
  return IPython.display.HTML(video_tag)


def plot_state_coverage(trajectories, shape):
  trajectories = np.array(trajectories).reshape(-1, *shape)

  fig, axs = plt.subplots(3, 3, figsize=(15, 10))
  # turn all axis off
  [axi.set_axis_off() for axi in axs.ravel()]

  def _plot_figure(dir, title, pos):
    if dir >= 0:
      dir_count = trajectories[np.sum((trajectories[:, 2] + trajectories[:, 0]) == 10 + dir, axis=(1, 2)).astype(bool)]
    else:
      dir_count = trajectories

    img = np.sum(np.array(dir_count)[:, 0] == 10, axis=0)
    a = axs[pos]
    im = a.imshow(img)
    fig.colorbar(im, ax=a)
    a.set_title(title.format(len(dir_count)))

  _plot_figure(-1, 'COMB-{}', (1, 1))
  _plot_figure(0, 'RIGHT-{}', (1, 2))
  _plot_figure(1, 'DOWN-{}', (2, 1))
  _plot_figure(2, 'LEFT-{}', (1, 0))
  _plot_figure(3, 'UP-{}', (0, 1))

  plt.show()
