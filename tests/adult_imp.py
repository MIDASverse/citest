from citest.data import adult
from MIDAS2 import model as md
import numpy as np

import cProfile
import pstats

import torch

np.random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":
    profiler = cProfile.Profile()

    data = adult(5000)

    train_index = np.random.choice(data.miss_data.shape[0], size=1800, replace=False)
    profiler.enable()
    midas_model = md.MIDAS()
    epochs = 250
    omit_first = True
    midas_model.fit(
        (data.miss_data.iloc[train_index, :].copy()),
        epochs=epochs,
        omit_first=omit_first,
        verbose=True,
        seed=42,
    )
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(50)
