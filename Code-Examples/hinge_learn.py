import numpy as np
import matplotlib.pyplot as plt

# 1) Generar dataset
np.random.seed(42)
N = 500
X = np.random.uniform(-5, 5, size=(N, 1))     # entradas
Y = np.maximum(0, X)                          # función hinge: y = max(0, x)

# 2) Inicializar pesos y biases
w1 = np.random.randn(1)   # peso capa entrada→oculta
b1 = np.random.randn(1)   # bias oculta
w2 = np.random.randn(1)   # peso capa oculta→salida
b2 = np.random.randn(1)   # bias salida

lr = 0.001    # learning rate
epochs = 2000

# 3) Entrenamiento por descenso de gradiente
for epoch in range(epochs):
    # --- Forward ---
    z1 = w1 * X + b1                   # pre-activación oculta
    h  = np.maximum(0, z1)             # ReLU
    y_pred = w2 * h + b2               # salida lineal

    # --- Cálculo de pérdida (MSE) ---
    loss = np.mean((y_pred - Y)**2)

    # --- Backward ---
    dL_dy = 2*(y_pred - Y) / N         # dL/d(y_pred)
    # capa salida
    dw2 = np.sum(dL_dy * h)
    db2 = np.sum(dL_dy)
    # backprop hacia oculta
    dh  = dL_dy * w2
    dz1 = dh * (z1 > 0)                # derivada ReLU
    dw1 = np.sum(dz1 * X)
    db1 = np.sum(dz1)

    # --- Update ---
    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

    # Opcional: imprimir cada 500 épocas
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}, loss = {loss:.4f}")

# 4) Visualizar ajuste final
plt.scatter(X, Y, alpha=0.3, label="Datos reales")
plt.scatter(X, y_pred, alpha=0.3, label="Predicción NN")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste de función hinge con NN pequeña")
plt.show()
