# Código Python para aprender forma de V con una NN de 1 entrada y 1 capa oculta de 2 neuronas

import numpy as np
import matplotlib.pyplot as plt

# 1) Generar dataset para función V (valor absoluto)
np.random.seed(42)
N = 500
X = np.random.uniform(-5, 5, size=(N, 1))       # entradas
Y = np.abs(X)                                   # función V: y = |x|

# 2) Inicializar pesos y biases
# Capa oculta: 2 neuronas + bias
w1 = np.random.randn(1, 2)   # pesos entrada→oculta (1×2)
b1 = np.random.randn(1, 2)   # bias capa oculta  (1×2)
# Capa de salida: 1 neurona + bias
w2 = np.random.randn(2, 1)   # pesos oculta→salida (2×1)
b2 = np.random.randn(1)      # bias salida       (1,)

# Hiperparámetros
lr = 0.001    # tasa de aprendizaje
epochs = 3000 # número de iteraciones

# Función ReLU y su derivada
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

# 3) Entrenamiento por descenso de gradiente
for epoch in range(epochs):
    # --- Forward pass ---
    z1 = X.dot(w1) + b1                # pre-activación oculta (N×2)
    h  = relu(z1)                      # activación oculta   (N×2)
    y_pred = h.dot(w2) + b2            # salida (N×1)

    # --- Pérdida MSE ---
    loss = np.mean((y_pred - Y)**2)

    # --- Backward pass ---
    dL_dy = 2*(y_pred - Y) / N         # gradiente salida (N×1)
    # Capa de salida
    dw2 = h.T.dot(dL_dy)               # gradientes w2 (2×1)
    db2 = np.sum(dL_dy, axis=0)        # gradiente b2 (1,)
    # Capa oculta
    dh  = dL_dy.dot(w2.T)              # (N×2)
    dz1 = dh * relu_deriv(z1)          # (N×2)
    dw1 = X.T.dot(dz1)                 # gradientes w1 (1×2)
    db1 = np.sum(dz1, axis=0)          # gradientes b1 (2,)

    # --- Actualización de parámetros ---
    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

# 4) Visualizar resultado
y_pred_final = relu(X.dot(w1) + b1).dot(w2) + b2

plt.figure(figsize=(6,4))
plt.scatter(X, Y, alpha=0.3, label="Datos reales")
plt.scatter(X, y_pred_final, alpha=0.3, label="Predicción NN")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste de función en V con NN pequeña")
plt.tight_layout()
plt.show()
