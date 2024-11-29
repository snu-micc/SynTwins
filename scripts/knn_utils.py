import logging
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Union
from functools import partial
import numpy as np

from sklearn.neighbors import BallTree
from .chem_utils import *

import logging
logger = logging.getLogger(__name__)

class MolEmbedder:
    def __init__(self, _nBits=256, useChirality=True, processes=1):
        self.processes = processes
        self.func = partial(get_fps, _radius=2, _nBits=_nBits, useChirality=useChirality)

    def _compute_mp(self, data):
        from pathos import multiprocessing as mp

        with mp.Pool(processes=self.processes) as pool:
            embeddings = pool.map(self.func, data)
        return embeddings

    def compute_embeddings(self, building_blocks):
#         logger.info(f"Will compute embedding with {self.processes} processes.")
        if self.processes == 1:
            embeddings = list(map(self.func, building_blocks))
        else:
            embeddings = self._compute_mp(building_blocks)
#         logger.info(f"Computed embeddings.")
        self.embeddings = embeddings
        return self

    def _save_npy(self, file: str):
        if self.embeddings is None:
            raise ValueError("Must have computed embeddings to save.")

        embeddings = np.asarray(self.embeddings)  # assume at least 2d
        np.save(file, embeddings)
        logger.info(f"Successfully saved data (shape={embeddings.shape}) to {file} .")
        return self

    def save_precomputed(self, file: str):
        """Saves pre-computed molecule embeddings to `*.npy`"""
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        if file.suffixes == [".npy"]:
            self._save_npy(file)
        else:
            raise NotImplementedError(f"File must have 'npy' extension, not {file.suffixes}")
        return self

    def _load_npy(self, file: Path):
        return np.load(file)

    def load_precomputed(self, file: str):
        """Loads a pre-computed molecule embeddings from `*.npy`"""
        file = Path(file)
        if file.suffixes == [".npy"]:
            self.embeddings = self._load_npy(file)
            self.kdtree = None
        else:
            raise NotImplementedError
        return self

    def init_balltree(self, metric='euclidean'):
        """Initializes a `BallTree`.

        Note:
            Can take a couple of minutes."""
        if self.embeddings is None:
            raise ValueError("Need emebddings to compute kdtree.")
        X = self.embeddings
        self.kdtree = BallTree(X, metric=metric)

        return self.kdtree