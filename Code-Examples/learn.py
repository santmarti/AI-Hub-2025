import numpy as np
import matplotlib.pyplot as plt

# (Optional) bump global font size
plt.rcParams.update({'font.size': 18})

# Parameters
alpha = 0.1
y = 1
x = 1
w_init = 0.0
steps = 40

# Compute trajectories
w_linear = [w_init]
w_perceptron = [w_init]
w1 = w2 = w_init
for _ in range(steps):
    w1 += alpha * y
    w_linear.append(w1)
    w2 += alpha * (y - x * w2)
    w_perceptron.append(w2)

# Plot
plt.figure(figsize=(8, 8))


color_lin = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]  # tab:blue (#1f77b4)
color_per = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]  # tab:orange (#ff7f0e)


# Transparent guiding lines
#plt.plot(w_linear, color=color_lin, alpha=0.2, linewidth=3)
plt.plot(w_perceptron, color=color_per, alpha=0.2, linewidth=3)

# Big markers with LaTeX labels (no \displaystyle)
#plt.scatter(range(steps + 1), w_linear, s=80, color=color_lin,
#            label=r'$w = w + \alpha\,y$')
plt.scatter(range(steps + 1), w_perceptron, s=80, color=color_per,
            label=r'$w = w + \alpha\,(y - xw)$'
            )

# Bold, LaTeX-mode axis labels & title
plt.xlabel(r'$\mathbf{Update\ Step}$', fontsize=22)
plt.ylabel(r'$\mathbf{w\ Value}$', fontsize=22)
plt.title(r'$\mathbf{Parameter\ Update\ Curves}$', fontsize=26)

plt.legend(fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()