import numpy as np
import matplotlib.pyplot as plt

# 1) Generar dataset para función V con ruido
np.random.seed(42)
N = 500
X = np.random.uniform(-5, 5, size=(N, 1))
noise_sigma = 0.2
Y = np.abs(X) + np.random.normal(0, noise_sigma, size=(N, 1))

# 2) Inicializar todos los pesos como w y biases
w1 = np.random.randn() * 0.1   # peso neurona oculta h1
w2 = np.random.randn() * 0.1   # peso neurona oculta h2
b1 = 0.0                       # bias neurona h1
b2 = 0.0                       # bias neurona h2

w3 = np.random.randn() * 0.1   # peso salida desde h1
w4 = np.random.randn() * 0.1   # peso salida desde h2
b3 = 0.0                       # bias de salida

# Hiperparámetros
lr = 0.01       # tasa de aprendizaje
epochs = 2000   # número de iteraciones

# ReLU y derivada
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

# 3) Entrenamiento por descenso de gradiente
for epoch in range(epochs):
    # Forward pass
    h1 = relu(w1 * X + b1)            # (N×1)
    h2 = relu(w2 * X + b2)            # (N×1)
    y_pred = w3 * h1 + w4 * h2 + b3   # (N×1)

    # Cálculo de pérdida MSE
    loss = np.mean((y_pred - Y)**2)

    # Backward pass
    dL = 2 * (y_pred - Y) / N         # gradiente de la salida (N×1)

    # Gradientes de la capa de salida
    dw3 = np.sum(dL * h1)
    dw4 = np.sum(dL * h2)
    db3_grad = np.sum(dL)

    # Gradientes de la capa oculta
    dh1 = dL * w3
    dh2 = dL * w4
    dz1 = dh1 * relu_deriv(w1 * X + b1)
    dz2 = dh2 * relu_deriv(w2 * X + b2)
    dw1_grad = np.sum(dz1 * X)
    dw2_grad = np.sum(dz2 * X)
    db1_grad = np.sum(dz1)
    db2_grad = np.sum(dz2)

    # Actualización de parámetros
    w1 -= lr * dw1_grad
    b1 -= lr * db1_grad
    w2 -= lr * dw2_grad
    b2 -= lr * db2_grad
    w3 -= lr * dw3
    w4 -= lr * dw4
    b3 -= lr * db3_grad

# 4) Visualizar resultado
y_pred_final = w3 * relu(w1 * X + b1) + w4 * relu(w2 * X + b2) + b3

plt.figure(figsize=(6,4))
plt.scatter(X, Y, alpha=0.3, label="Datos reales con ruido")
#plt.scatter(X, y_pred_final, alpha=0.3, label="Predicción NN")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste de |x| con NN (pesos w1–w4)")
plt.tight_layout()


print("Parámetros finales aprendidos:")
print(f"w1 (peso h1): {w1:.4f}, b1: {b1:.4f}")
print(f"w2 (peso h2): {w2:.4f}, b2: {b2:.4f}")
print(f"w3 (peso salida desde h1): {w3:.4f}")
print(f"w4 (peso salida desde h2): {w4:.4f}")
print(f"b3 (bias salida): {b3:.4f}")


plt.show()


