import numpy as np
import matplotlib.pyplot as plt

# Activation functions and their derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1 - np.tanh(z)**2

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

# Map activation names to functions
activations = {
    'sigmoid': (sigmoid, sigmoid_prime),
    'tanh':    (tanh, tanh_prime),
    'relu':    (relu, relu_prime)
}

def train_gate(gate_name, X, y, activation='sigmoid', epochs=10000, eta=0.1):
    """
    Train a 2-layer neural net on a logic gate dataset.
    
    Parameters:
    - gate_name: label for plotting
    - X: inputs, shape (N, 2)
    - y: targets, shape (N,)
    - activation: one of 'sigmoid', 'tanh', 'relu'
    - epochs: number of training epochs
    - eta: learning rate
    """
    # Get activation functions
    act, act_prime = activations[activation]
    
    # Prepare data with bias
    X_bias = np.vstack([X.T, np.ones((1, X.shape[0]))])  # (3, N)
    y = y.reshape(1, -1)  # (1, N)
    
    # Initialize weights
    W0 = np.random.randn(2, 3)
    W1 = np.random.randn(1, 3)
    
    losses = []
    N = X.shape[0]
    
    for epoch in range(epochs):
        # Forward pass
        z0 = W0 @ X_bias           # (2, N)
        h = act(z0)                # (2, N)
        h_aug = np.vstack([h, np.ones((1, N))])  # (3, N)
        z1 = W1 @ h_aug            # (1, N)
        y_pred = act(z1)           # (1, N)
        
        # Loss (MSE)
        loss = np.mean(0.5 * (y_pred - y)**2)
        losses.append(loss)
        
        # Backpropagation
        delta1 = (y_pred - y) * act_prime(z1)                # (1, N)
        grad_W1 = delta1 @ h_aug.T / N                       # (1, 3)
        
        W1_no_bias = W1[:, :2]                               # (1, 2)
        delta0 = (W1_no_bias.T @ delta1) * act_prime(z0)     # (2, N)
        grad_W0 = delta0 @ X_bias.T / N                      # (2, 3)
        
        # Gradient descent updates
        W1 -= eta * grad_W1
        W0 -= eta * grad_W0
    
    # Plot loss curve
    plt.figure()
    plt.plot(losses)
    plt.title(f"Loss Curve for {gate_name.upper()} Gate ({activation})")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.show()
    
    # Final predictions
    z0 = W0 @ X_bias
    h = act(z0)
    h_aug = np.vstack([h, np.ones((1, N))])
    z1 = W1 @ h_aug
    y_final = act(z1)
    
    print(f"\nFinal outputs for {gate_name.upper()} gate using {activation}:")
    for inp, out in zip(X, y_final.ravel()):
        print(f"  Input {inp} -> {out:.4f}")

# Logic gate datasets
gates = {
    'and': (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0, 0, 0, 1])),
    'or' : (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0, 1, 1, 1])),
    'xor': (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0, 1, 1, 0]))
}

# Example: train XOR gate with each activation
for act_name in ['sigmoid', 'tanh', 'relu']:
    X, y = gates['xor']
    train_gate('xor', X, y, activation=act_name, epochs=5000, eta=0.1)