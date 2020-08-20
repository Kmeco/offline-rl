import numpy as np
import copy
import matplotlib.pyplot as plt


base = np.array([[[2., 2., 2., 2., 2., 2.],
                  [2., 1., 1., 1., 1., 2.],
                  [2., 1., 1., 1., 1., 2.],
                  [2., 1., 1., 1., 1., 2.],
                  [2., 1., 1., 1., 8., 2.],
                  [2., 2., 2., 2., 2., 2.]],

                 [[5., 5., 5., 5., 5., 5.],
                  [5., 0., 0., 0., 0., 5.],
                  [5., 0., 0., 0., 0., 5.],
                  [5., 0., 0., 0., 0., 5.],
                  [5., 0., 0., 0., 1., 5.],
                  [5., 5., 5., 5., 5., 5.]],

                 [[0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.]]], dtype=np.float32)


def get_full_observation(state):
  obs = copy.copy(base)
  obs[0, state[0], state[1]] = 10
  obs[2, state[0], state[1]] = state[2]
  return obs


def evaluate_q(critic_network):
  """
  Q is an array of size (d, n, n, a) where:
  d is the number of directions: 4
  n is the size of the grid: 6
  a is a number of actions: 3
  """
  Q = np.zeros((4, 6, 6, 3))
  for i in range(1, 5):
    for j in range(1, 5):
      state = copy.copy(base)
      state[0, i, j] = 10
      for k in range(4):
        state = (i, j, k)
        obs = get_full_observation(state)
        Q[k, i, j] = critic_network(obs.reshape(1, -1)).numpy()
  return Q


map_from_int_to_dir = lambda a: (r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$", r"$\uparrow$")[a]
map_from_action_to_name = lambda a, d: [r"$\circlearrowleft$", r"$\circlearrowright$", map_from_int_to_dir(d)][a]


def plot_values(values, colormap='pink', vmin=0.8, vmax=1):
  plt.imshow(values, interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax)
  plt.yticks([])
  plt.xticks([])
  plt.colorbar(ticks=[vmin, vmax])


def plot_state_value(action_values):
  q = action_values
  vmin = np.min(action_values[action_values != 0])
  vmax = np.max(action_values)
  v = 0.9 * np.max(q, axis=-1) + 0.1 * np.mean(q, axis=-1)
  plot_values(v, colormap='summer', vmin=vmin, vmax=vmax)
  plt.title("$v(s)$")


def plot_grid(obs):
  obs = obs.reshape(3, 6, 6)
  layout = (obs[0] == 2).astype(int) * (-1)
  plt.imshow(layout > -1, interpolation="nearest", cmap="YlOrRd_r")
  ax = plt.gca()
  ax.grid(0)
  plt.xticks([])
  plt.yticks([])
  goal = (obs[0] == 8).astype(int)
  plt.text(
    np.argmax(np.max(goal, axis=1)), np.argmax(np.max(goal, axis=0)),
    r"$\mathbf{G}$", ha='center', va='center', size='large')
  h, w = layout.shape
  for y in range(h - 3):
    plt.plot([+0.5, w - 1.5], [y + 1.5, y + 1.5], '-k', lw=2, alpha=0.5)
  for x in range(w - 3):
    plt.plot([x + 1.5, x + 1.5], [+0.5, h - 1.5], '-k', lw=2, alpha=0.5)


def plot_greedy_policy(obs, q, dir):
  action_names = lambda a, d: [r"$\circlearrowleft$", r"$\circlearrowright$", map_from_int_to_dir(d)][a]
  greedy_actions = np.argmax(q, axis=2)
  plot_grid(obs)
  plt.title(f"The grid: {map_from_int_to_dir(dir)}", size='large')
  for i in range(1, 5):
    for j in range(1, 5):
      action_name = action_names(greedy_actions[i, j], dir)
      if (i != 4 or j != 4):
        plt.text(j, i, action_name, ha='center', va='center', size='large')


def visualize_policy(action_values, obs):
  q = action_values
  fig = plt.figure(figsize=(17, 12))
  fig.subplots_adjust(wspace=0.3, hspace=0.3)
  vmin = np.min(action_values[action_values != 0])
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
    plot_greedy_policy(obs, q[row], row)
  return plt

#   for ax, row in zip(axes[:,0], rows):
#     ax.set_ylabel(row, rotation=0, size='large')