import numpy as np
import matplotlib.pyplot as plt

# Configuración
np.random.seed(42)
N = 500                     # número de muestras
hidden_neurons = 20          # neuronas en capa oculta
lr = 0.001                 # tasa de aprendizaje
epochs = 9000               # iteraciones

# 1) Generar dataset para la parábola y = x^2
X = np.random.uniform(-10, 10, (N, 1))    # entradas (N×1)
noise_sigma = 3
Y = X**2 + np.random.normal(0, noise_sigma, size=(N, 1))

# 2) Inicializar pesos y biases
W1 = np.random.randn(1, hidden_neurons) * 0.1  # pesos entrada→oculta
b1 = np.zeros((1, hidden_neurons))             # bias capa oculta

W2 = np.random.randn(hidden_neurons, 1) * 0.1  # pesos oculta→salida
b2 = np.zeros((1, 1))                          # bias salida

# Definición de ReLU y su derivada
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

# 3) Entrenamiento por descenso de gradiente
for epoch in range(epochs):
    # Forward pass
    Z1 = X.dot(W1) + b1       # (N×hidden_neurons)
    H  = relu(Z1)             # activación oculta (N×hidden_neurons)
    Y_pred = H.dot(W2) + b2   # salida predicha (N×1)

    # Cálculo de pérdida MSE
    loss = np.mean((Y_pred - Y)**2)

    # Backward pass
    dL_dy = 2 * (Y_pred - Y) / N          # (N×1)
    # Gradientes salida
    dW2 = H.T.dot(dL_dy)                  # (hidden_neurons×1)
    db2 = np.sum(dL_dy, axis=0, keepdims=True)  # (1×1)

    # Gradientes capa oculta
    dH = dL_dy.dot(W2.T)                  # (N×hidden_neurons)
    dZ1 = dH * relu_deriv(Z1)             # (N×hidden_neurons)
    dW1 = X.T.dot(dZ1)                    # (1×hidden_neurons)
    db1 = np.sum(dZ1, axis=0, keepdims=True)    # (1×hidden_neurons)

    # Actualización de parámetros
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # Mostrar pérdida cada 1000 épocas
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, loss = {loss:.6f}")

# 4) Visualizar resultado
Y_pred_final = relu(X.dot(W1) + b1).dot(W2) + b2

plt.figure(figsize=(6, 4))
plt.scatter(X, Y, alpha=0.3, label="Datos reales (y = x^2)")
plt.scatter(X, Y_pred_final, alpha=0.3, label="Predicción NN")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste de y = x^2 con NN usando ReLU")
plt.legend()
plt.tight_layout()


# 5) Imprimir parámetros aprendidos
print("")
print("")
print("Parámetros finales aprendidos:")
print("W1 (entrada→oculta):")
print(W1)
print("\nb1 (bias capa oculta):")
print(b1)
print("\nW2 (oculta→salida):")
print(W2)
print("\nb2 (bias salida):")
print(b2)







plt.show()
