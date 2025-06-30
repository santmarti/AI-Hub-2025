import numpy as np
import matplotlib.pyplot as plt

# Sigmoid and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def train_xor_improved(epochs=10000, eta=0.1, momentum=0.9):
    # XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0, 1, 1, 0]])  # shape (1,4)
    
    # Add bias term to inputs: shape (3,4)
    X_bias = np.vstack([X.T, np.ones((1, X.shape[0]))])
    
    # Xavier initialization
    fan_in_hidden = X_bias.shape[0]  # 3
    fan_out_hidden = 2
    W0 = np.random.randn(fan_out_hidden, fan_in_hidden) * np.sqrt(1 / fan_in_hidden)
    fan_in_output = W0.shape[0] + 1  # include bias node
    fan_out_output = 1
    W1 = np.random.randn(fan_out_output, fan_in_output) * np.sqrt(1 / fan_in_output)
    
    # Momentum variables
    vW0 = np.zeros_like(W0)
    vW1 = np.zeros_like(W1)
    
    losses = []
    N = X.shape[0]
    
    for epoch in range(epochs):
        # Forward pass
        z0 = W0 @ X_bias                # (2,4)
        h = sigmoid(z0)                 # (2,4)
        h_aug = np.vstack([h, np.ones((1, N))])  # (3,4)
        z1 = W1 @ h_aug                 # (1,4)
        y_pred = sigmoid(z1)            # (1,4)
        
        # Cross-entropy loss
        eps = 1e-8
        loss = -np.mean(y * np.log(y_pred + eps) + (1-y) * np.log(1 - y_pred + eps))
        losses.append(loss)
        
        # Gradients (sigmoid+CE => delta1 = y_pred - y)
        delta1 = (y_pred - y)                       # (1,4)
        grad_W1 = (delta1 @ h_aug.T) / N            # (1,3)
        
        # Backprop to hidden
        W1_no_bias = W1[:, :2]                      # (1,2)
        delta0 = (W1_no_bias.T @ delta1) * sigmoid_prime(z0)  # (2,4)
        grad_W0 = (delta0 @ X_bias.T) / N           # (2,3)
        
        # Momentum updates
        vW1 = momentum * vW1 - eta * grad_W1
        W1 += vW1
        
        vW0 = momentum * vW0 - eta * grad_W0
        W0 += vW0
    
    # Plot loss
    plt.figure()
    plt.plot(losses)
    plt.title("Improved XOR Training (CE + Xavier + Momentum)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.show()
    
    # Final outputs
    z0 = W0 @ X_bias
    h = sigmoid(z0)
    h_aug = np.vstack([h, np.ones((1, N))])
    z1 = W1 @ h_aug
    y_final = sigmoid(z1)
    
    print("Final XOR predictions:")
    for inp, out in zip(X, y_final.ravel()):
        print(f" {inp} -> {out:.4f}")

# Run improved XOR training
train_xor_improved()