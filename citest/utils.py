import numpy as np


def BCEclip(
    p: np.array,
    y: np.array,
) -> float:
    """Return mean binary cross-entropy loss with clipping

    Avoids numerical instability when taking the log of very small numbers.

    Note, numpy's divide by zero warning is silenced.

    Args:
        p: Predicted probabilities
        y: True binary labels

    Returns:
        Clipped binary cross-entropy loss value for input

    Raises:
        None

    """
    with np.errstate(divide="ignore"):
        clipped = np.mean(
            -(
                y * np.clip(np.log(p), a_min=-100, a_max=None)
                + (1 - y) * np.clip(np.log(1 - p), a_min=-100, a_max=None)
            )
        )

    return clipped
