# AI_From_Scratch
Implementation of a neural network from scratch using only NumPy, trained on MNIST dataset

## Installation
1. Clone the repository
```bash
   git clone https://github.com/tcjordan3/AI_From_Scratch.git
   cd AI_From_Scratch
```

2. Create a virtual environment (Python 3.9-3.11 required)
```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows Git Bash
   # or
   venv\Scripts\activate  # On Windows CMD
```

3. Install PyTorch (CPU version)
```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

4. Install the package
```bash
   pip install -e .
```

## Usage
```bash
  python src/training_loop.py
```
