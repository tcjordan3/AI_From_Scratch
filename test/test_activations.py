import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from activation_functions import ReLU, SoftmaxCrossEntropy

def test_relu():
    """
    Validate ReLU implementation against PyTorch
    """

    print("=" * 50)
    print("RELU TEST")
    print("=" * 50)
    
    # Create test input
    batch_size = 4
    dim = 6
    Z_np = np.random.randn(batch_size, dim).astype(np.float32)
    Z_torch = torch.tensor(Z_np, requires_grad=True)
    
    # Custom ReLU implementation
    relu = ReLU()
    A_custom = relu.forward(Z_np)
    
    # PyTorch implementation
    A_torch = F.relu(Z_torch)
    A_torch_np = A_torch.detach().numpy()
    
    # Compare forward pass
    forward_diff = np.abs(A_custom - A_torch_np).max()
    print(f"Forward pass max difference: {forward_diff:.2e}")
    print(f"Forward pass: {'✓ PASS' if forward_diff < 1e-6 else '✗ FAIL'}")
    
    # Test backward pass
    dA_np = np.random.randn(batch_size, dim).astype(np.float32)
    dA_torch = torch.tensor(dA_np)
    
    # Custom backward pass
    dZ_custom = relu.backward(dA_np)
    
    # PyTorch backward pass
    A_torch.backward(dA_torch)
    dZ_torch_np = Z_torch.grad.numpy()
    
    # Compare backward pass
    backward_diff = np.abs(dZ_custom - dZ_torch_np).max()
    print(f"Backward pass max difference: {backward_diff:.2e}")
    print(f"Backward pass: {'✓ PASS' if backward_diff < 1e-6 else '✗ FAIL'}")
    print()
    
    return forward_diff < 1e-6 and backward_diff < 1e-6

def test_softmax_cross_entropy():
    """
    Validate SoftmaxCrossEntropy implementation against PyTorch
    """

    print("=" * 50)
    print("SOFTMAX + CROSS-ENTROPY TEST")
    print("=" * 50)
    
    # Create test input
    batch_size = 8
    num_classes = 10
    Z_np = np.random.randn(batch_size, num_classes).astype(np.float32)
    Z_torch = torch.tensor(Z_np, requires_grad=True)
    
    # Create one-hot labels
    labels = np.random.randint(0, num_classes, size=batch_size)
    Y_np = np.zeros((batch_size, num_classes), dtype=np.float32)
    Y_np[np.arange(batch_size), labels] = 1.0
    
    # Custom implementation
    softmax_ce = SoftmaxCrossEntropy()
    loss_custom = softmax_ce.forward(Z_np, Y_np)
    
    # PyTorch implementation
    loss_torch = F.cross_entropy(Z_torch, torch.tensor(labels, dtype=torch.long))
    loss_torch_value = loss_torch.item()
    
    # Compare forward pass (loss)
    loss_diff = abs(loss_custom - loss_torch_value)
    print(f"Loss difference: {loss_diff:.2e}")
    print(f"Your loss: {loss_custom:.6f}")
    print(f"PyTorch loss: {loss_torch_value:.6f}")
    print(f"Forward pass: {'✓ PASS' if loss_diff < 1e-5 else '✗ FAIL'}")
    
    # Test backward pass
    dZ_custom = softmax_ce.backward()
    
    # PyTorch backward
    loss_torch.backward()
    dZ_torch_np = Z_torch.grad.numpy()
    
    # Compare backward pass
    backward_diff = np.abs(dZ_custom - dZ_torch_np).max()
    print(f"Backward pass max difference: {backward_diff:.2e}")
    print(f"Backward pass: {'✓ PASS' if backward_diff < 1e-6 else '✗ FAIL'}")
    print()
    
    return loss_diff < 1e-5 and backward_diff < 1e-6

def test_softmax_probabilities():
    """
    Additional test: verify softmax outputs are valid probabilities
    """
    
    print("=" * 50)
    print("SOFTMAX PROPERTIES TEST")
    print("=" * 50)
    
    batch_size = 5
    num_classes = 10
    Z_np = np.random.randn(batch_size, num_classes).astype(np.float32)
    Y_np = np.eye(num_classes)[np.random.randint(0, num_classes, batch_size)]
    
    softmax_ce = SoftmaxCrossEntropy()
    _ = softmax_ce.forward(Z_np, Y_np)
    
    # Check properties
    probs = softmax_ce.A
    
    # All probabilities should be in [0, 1]
    all_valid = np.all((probs >= 0) & (probs <= 1))
    print(f"All probabilities in [0, 1]: {'✓ PASS' if all_valid else '✗ FAIL'}")
    
    # Each row should sum to ~1
    row_sums = np.sum(probs, axis=1)
    sums_correct = np.allclose(row_sums, 1.0, atol=1e-6)
    print(f"Row sums equal 1: {'✓ PASS' if sums_correct else '✗ FAIL'}")
    print(f"Row sums: {row_sums}")
    print()
    
    return all_valid and sums_correct

if __name__ == "__main__":
    relu_pass = test_relu()
    softmax_pass = test_softmax_cross_entropy()
    properties_pass = test_softmax_probabilities()
    
    print("=" * 50)
    all_pass = relu_pass and softmax_pass and properties_pass
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print("=" * 50)