import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn

from layer import Layer

def test_layer_forward_backward():
    """
    Validate Layer implementation against PyTorch
    """
    
    batch_size = 4
    input_dim = 10
    output_dim = 5
    
    # Custom layer implementation
    my_layer = Layer(input_dim, output_dim)
    
    # PyTorch layer with same weights/biases
    torch_layer = nn.Linear(input_dim, output_dim)
    torch_layer.weight.data = torch.tensor(my_layer.W.T, dtype=torch.float32)  # PyTorch uses transposed weights
    torch_layer.bias.data = torch.tensor(my_layer.b, dtype=torch.float32)
    
    # Create random input
    X_np = np.random.randn(batch_size, input_dim).astype(np.float32)
    X_torch = torch.tensor(X_np, requires_grad=True)
    
    print("=" * 50)
    print("FORWARD PASS TEST")
    print("=" * 50)
    
    # Forward pass - custom implementation
    Z_yours = my_layer.forward(X_np)
    
    # Forward pass - PyTorch
    Z_torch = torch_layer(X_torch)
    Z_torch_np = Z_torch.detach().numpy()
    
    # Compare outputs
    forward_diff = np.abs(Z_yours - Z_torch_np).max()
    print(f"Max difference in forward pass: {forward_diff:.2e}")
    print(f"Forward pass match: {'✓ PASS' if forward_diff < 1e-6 else '✗ FAIL'}")
    
    print("\n" + "=" * 50)
    print("BACKWARD PASS TEST")
    print("=" * 50)
    
    # Create gradient signal (pretend this came from loss)
    dZ_np = np.random.randn(batch_size, output_dim).astype(np.float32)
    dZ_torch = torch.tensor(dZ_np)
    
    # Backward pass - custom implementation
    dX_yours = my_layer.backward(dZ_np)
    
    # Backward pass - PyTorch
    Z_torch.backward(dZ_torch)
    dX_torch_np = X_torch.grad.numpy()
    dW_torch_np = torch_layer.weight.grad.T.numpy()  # Transpose back
    db_torch_np = torch_layer.bias.grad.numpy()
    
    # Compare gradients
    dX_diff = np.abs(dX_yours - dX_torch_np).max()
    dW_diff = np.abs(my_layer.dW - dW_torch_np).max()
    db_diff = np.abs(my_layer.db - db_torch_np).max()
    
    print(f"Max difference in dX: {dX_diff:.2e}")
    print(f"Max difference in dW: {dW_diff:.2e}")
    print(f"Max difference in db: {db_diff:.2e}")
    
    print(f"\ndX match: {'✓ PASS' if dX_diff < 1e-6 else '✗ FAIL'}")
    print(f"dW match: {'✓ PASS' if dW_diff < 1e-6 else '✗ FAIL'}")
    print(f"db match: {'✓ PASS' if db_diff < 1e-6 else '✗ FAIL'}")
    
    print("\n" + "=" * 50)
    all_pass = all([forward_diff < 1e-6, dX_diff < 1e-6, dW_diff < 1e-6, db_diff < 1e-6])
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print("=" * 50)

if __name__ == "__main__":
    test_layer_forward_backward()