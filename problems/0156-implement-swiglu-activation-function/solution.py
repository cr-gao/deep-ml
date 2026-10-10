import numpy as np

def silu(x: np.ndarray) -> np.ndarray:
    return x * (1 / (1 + np.exp(-x)))

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Your code here
    d = x.shape[1] // 2
    gate = silu(x[:, d:])
    scores = gate * x[:, :d]

    return scores