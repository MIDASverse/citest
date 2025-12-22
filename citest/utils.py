import numpy as np


def BCEclip(
    p: np.array,
    y: np.array,
) -> float:
    """Return observation-level binary cross-entropy loss with clipping

    Avoids numerical instability when taking the log of very small numbers.

    Note, numpy's divide by zero warning is silenced.

    Args:
        p: Predicted probabilities
        y: True binary labels
        sample_weight: array-like, same shape or broadcastable, to weight entries

    Returns:
        Clipped binary cross-entropy loss value for input

    Raises:
        None

    """
    assert p.shape == y.shape, "p and y must have the same shape"

    eps = 1e-15
    p_clipped = np.clip(p, eps, 1 - eps)

    with np.errstate(divide="ignore"):
        clipped = -(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
    return clipped
