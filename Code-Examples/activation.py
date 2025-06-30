import numpy as np
import matplotlib.pyplot as plt

# Define the range for z
z = np.linspace(-10, 10, 400)

# Compute sigmoid and its derivative
sigmoid = 1 / (1 + np.exp(-z))
sigmoid_derivative = sigmoid * (1 - sigmoid)

# Create a square figure
plt.figure(figsize=(8, 6))
plt.plot(z, sigmoid, linewidth=4, label='sigmoid(z)')  # línea gruesa
#plt.plot(z, sigmoid_derivative, linewidth=4, label="sigmoid'(z)")  # línea gruesa
plt.title('Función Sigmoide', fontsize=24)  # título con fuente grande
plt.xlabel('x', fontsize=18)                  # etiqueta eje x con fuente grande
plt.ylabel('σ(x)', fontsize=18)               # etiqueta eje y con fuente grande
plt.xticks(fontsize=14)                       # ticks eje x con fuente grande
plt.yticks(fontsize=14) 

plt.legend()
plt.grid(True)

# Display the plot
plt.show()



# Define the range for z
z = np.linspace(-2, 2, 400)

# Compute ReLU and its derivative
relu = np.maximum(0, z)
relu_derivative = np.where(z > 0, 1, 0)

# Create a square figure
plt.figure(figsize=(6, 6))

# Plot ReLU and its derivative
plt.plot(z, relu, label='ReLU(z)')
plt.plot(z, relu_derivative, label="ReLU'(z)")

# Labeling
plt.xlabel('z')
plt.ylabel('Function value')
plt.title('ReLU and Its Derivative')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()