import numpy as np
import matplotlib.pyplot as plt

# 1) Generate realistic height data
np.random.seed(0)
n_samples = 50
ages = np.random.uniform(0, 70, n_samples)
ages.sort()
heights = 50 + 120*(1 - np.exp(-0.04*ages)) + np.random.normal(0, 5, size=n_samples)

# 2) Normalize ages (feature scaling)
age_mean = ages.mean()
age_std = ages.std()
ages_norm = (ages - age_mean) / age_std

# 3) SGD hyperparameters
lr = 1e-1       # higher learning rate now stable thanks to normalization
n_steps = 500  # number of SGD updates

# 4) Initialize parameters
w0, w1 = 0.0, 0.0

# 5) Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(ages, heights, color='blue', label='Data')
x_line = np.array([0, 80])

# 6) Stochastic gradient descent with normalized input
for _ in range(n_steps):
    j = np.random.randint(n_samples)
    x_j_norm = ages_norm[j]
    y_j = heights[j]
    pred = w0 + w1 * x_j_norm
    error = y_j - pred
    # update bias and slope
    w0 += lr * error
    w1 += lr * error * x_j_norm
    # plot intermediate fit (convert back to original age scale)
    x_norm_line = (x_line - age_mean) / age_std
    plt.plot(x_line, w0 + w1 * x_norm_line, color='red', alpha=0.02)

# 7) Final fitted line
x_norm_line = (x_line - age_mean) / age_std
plt.plot(x_line, w0 + w1 * x_norm_line,
         color='red', linewidth=2,
         label=f'Fit: height = {w0:.1f} + {w1:.2f}·((age-{age_mean:.1f})/{age_std:.1f})')

# 8) Fixed axis limits
plt.xlim(0, 80)
plt.ylim(0, 200)

# 9) Styling
plt.xlabel('Age (years)', fontsize=16)
plt.ylabel('Height (cm)', fontsize=16)
plt.title('Height vs Age: SGD with Normalized Age Feature', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()