import numpy as np
import matplotlib.pyplot as plt

# Increase global font size for better readability
plt.rcParams.update({'font.size': 18})

# Learning rate and dynamic trial counts
alpha = 0.05
trials_s1 = 30  # example: change this to any number of trials for scenario 1
trials_s2 = 30  # example: change this to any number of trials for scenario 2

# Define inputs for both scenarios
# Scenario 1: first half [1,0], second half [1,1]
inputs_s1 = [np.array([1, 0])] * (trials_s1 // 2) + [np.array([1, 1])] * (trials_s1 - trials_s1 // 2)
# Scenario 2: all [1,1]
inputs_s2 = [np.array([1, 1])] * trials_s2
y = 1

def run_updates(inputs):
    w = np.zeros(2)
    traj = [w.copy()]
    for x in inputs:
        w += alpha * (y - np.dot(x, w)) * x
        traj.append(w.copy())
    return np.array(traj)

# Compute trajectories
traj_s1 = run_updates(inputs_s1)
traj_s2 = run_updates(inputs_s2)

# Colors for w0 and w1
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_w0, color_w1 = colors[0], colors[1]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Scenario 1 plot (no jitter)
n1 = len(inputs_s1)
ax1.plot(traj_s1[:, 0], color=color_w0, alpha=0.2, linewidth=3)
ax1.plot(traj_s1[:, 1], color=color_w1, alpha=0.2, linewidth=3)
ax1.scatter(range(n1 + 1), traj_s1[:, 0], s=80, color=color_w0, marker='o', label=r'$w_0$')
ax1.scatter(range(n1 + 1), traj_s1[:, 1], s=80, color=color_w1, marker='x', label=r'$w_1$')
ax1.set_xlabel(r'$\mathbf{Update\ Number}$', fontsize=22)
ax1.set_ylabel(r'$\mathbf{Weight\ Value}$', fontsize=22)
ax1.set_title(r'$x=[1,0]$ and then $x=[1,1]$', fontsize=22)
ax1.legend(fontsize=16)
ax1.set_xticks(range(0, n1 + 1, max(1, n1 // 5)))
ax1.tick_params(labelsize=18)
ax1.grid(True)

# Scenario 2 plot (with jitter on w1)
n2 = len(inputs_s2)
ax2.plot(traj_s2[:, 0], color=color_w0, alpha=0.2, linewidth=3)
ax2.plot(traj_s2[:, 1], color=color_w1, alpha=0.2, linewidth=3)
ax2.scatter(range(n2 + 1), traj_s2[:, 0], s=80, color=color_w0, marker='o', label=r'$w_0$')
jitter = 0.1 * (np.random.rand(n2 + 1) - 0.5)
ax2.scatter(np.arange(n2 + 1) + jitter, traj_s2[:, 1],
            s=80, color=color_w1, marker='x', label=r'$w_1$')
ax2.set_xlabel(r'$\mathbf{Update\ Number}$', fontsize=22)
ax2.set_ylabel(r'$\mathbf{Weight\ Value}$', fontsize=22)
ax2.set_title(r'all $x=[1,1]$', fontsize=22)
ax2.legend(fontsize=16)
ax2.set_xticks(range(0, n2 + 1, max(1, n2 // 5)))
ax2.tick_params(labelsize=18)
ax2.grid(True)

plt.tight_layout()
plt.show()