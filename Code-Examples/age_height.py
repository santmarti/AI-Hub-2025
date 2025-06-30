import numpy as np
import matplotlib.pyplot as plt

# 1) Generate realistic height data
np.random.seed(0)
n_samples = 50
ages = np.random.uniform(0, 70, n_samples)
ages.sort()
heights = 50 + 120*(1 - np.exp(-0.04*ages)) + np.random.normal(0, 5, size=n_samples)

# 2) SGD hyperparameters
lr = 1e-4       # ↑ higher learning rate
n_steps = 500  # fewer steps since updates are larger

# 3) Initialize parameters
w0, w1 = 0.0, 0.0

# 4) Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(ages, heights, c='blue', label='Data')
plt.tick_params(labelsize=18)
x_line = np.array([0, 80])

# 5) Stochastic gradient descent loop
for _ in range(n_steps):
    j = np.random.randint(n_samples)
    x_j, y_j = ages[j], heights[j]
    error = y_j - (w0 + w1 * x_j)
    # SGD updates
    w0 += lr * error
    w1 += lr * error * x_j
    # draw intermediate fit
    plt.plot(x_line, w0 + w1 * x_line, color='red', alpha=0.02)

# 6) Final fit
plt.plot(x_line, w0 + w1 * x_line,
         color='red', linewidth=2,
         label=f'Fit: height = {w0:.1f} + {w1:.2f}·age')

# 7) Fixed axes
plt.xlim(0, 70)
plt.ylim(0, 200)

# 8) Styling
plt.xlabel('Age (years)', fontsize=16)
plt.ylabel('Height (cm)', fontsize=16)
#plt.title('Height vs Age: Stochastic GD Fit (lr=1e-3)', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()