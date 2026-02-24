import numpy as np


def BCEclip(
    p: np.array,
    y: np.array,
) -> float:
    """Element-wise binary cross-entropy with clipped probabilities.

    Parameters
    ----------
    p : np.ndarray
        Predicted probabilities.
    y : np.ndarray
        True binary labels (same shape as p).

    Returns
    -------
    np.ndarray
        Per-element BCE loss (same shape as input).
    """
    assert p.shape == y.shape, "p and y must have the same shape"

    eps = 1e-15
    p_clipped = np.clip(p, eps, 1 - eps)

    with np.errstate(divide="ignore"):
        clipped = -(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
    return clipped
