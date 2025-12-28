import numpy as np
import logging
import torchvision.datasets as datasets
import math

from network import Network
from optimizer import SGD

"""
Driving training loop module. Handles loading the dataset, and defines the training loop for our custom Neural Network
"""

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def load():
    """
    Load and preprocess MNINST dataset via torchvision

    returns:
        x_train - training data
        x_validate - validation data
        x_test - test data
        y_train - training labels
        y_validate - validation labels
        y_test - test labels
    """

    train_dataset = datasets.MNIST(root='../data', train=True, download=True)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True)

    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    # Flatten images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Normalize pixel values
    x_train = np.divide(x_train, 255.0).astype(np.float32)
    x_test = np.divide(x_test, 255.0).astype(np.float32)

    # One-hot encode labels
    y_train_onehot = np.zeros((y_train.shape[0], 10), dtype=np.float32)
    y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

    y_test_onehot = np.zeros((y_test.shape[0], 10), dtype=np.float32)
    y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1

    # Split validation set
    x_train, x_validate = x_train[:50000], x_train[50000:]
    y_train, y_validate, y_test = y_train_onehot[:50000], y_train_onehot[50000:], y_test_onehot

    logging.info(f"Training data shape: {x_train.shape}")
    logging.info(f"Validation data shape: {x_validate.shape}")
    logging.info(f"Testing data shape: {x_test.shape}")
    logging.info(f"Training labels shape: {y_train.shape}")
    logging.info(f"Validation labels shape: {y_validate.shape}")
    logging.info(f"Testing labels shape: {y_test.shape}")

    return x_train, x_validate, x_test, y_train, y_validate, y_test

def batch(X, Y, N, shuffle=True):
    """
    Yield batches of size N from the dataset for training

    args:
        X - data
        Y - labels
        N - batch size
        shuffle - if true, shuffle data indices

    yields:
        X_batch - batched training data
        Y_batch - batched labels
    """

    indices = np.arange(len(X))

    # Optionally shuffle indices
    if shuffle:
        np.random.shuffle(indices)
        
    # Generate batch
    for i_start in range(0, len(X), N):
        i_end = i_start + N

        i_batch = indices[i_start:i_end]
        X_batch = X[i_batch]
        Y_batch = Y[i_batch]

        yield X_batch, Y_batch

def train():
    """
    Training loop entrypoint
    """

    # Load dataset
    X_train, X_validate, X_test, Y_train, Y_validate, Y_test = load()

    # Define network
    nn = Network()
    optimizer = SGD(nn.get_parameters(), LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0
        batches = 0

        # Training
        for X_batch, Y_batch in batch(X_train, Y_train, BATCH_SIZE):
            L = nn.forward(X_batch, Y_batch)
            nn.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += L
            batches += 1

        avg_loss = total_loss/batches

        # Validation
        L = nn.forward(X_validate, Y_validate)
        A = nn.forward(X_validate, Y=None)
        
        predicted = np.argmax(A, axis=1)
        correct = (predicted == np.argmax(Y_validate, axis=1)).sum()
        accuracy = 100*correct/len(Y_validate)
        
        # Log progress
        logging.info(f"Epoch {epoch+1}/{EPOCHS}")
        logging.info(f"Train Loss: {avg_loss:.4f}")
        logging.info(f"Val Loss: {L:.4f}")
        logging.info(f"Val Accuracy: {accuracy:.2f}%")

    # Testing
    A_test = nn.forward(X_test, Y=None)
    test_predicted = np.argmax(A_test, axis=1)
    test_correct = (test_predicted == np.argmax(Y_test, axis=1)).sum()
    test_accuracy = 100*test_correct/len(Y_test)
    logging.info(f"Final Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    train()