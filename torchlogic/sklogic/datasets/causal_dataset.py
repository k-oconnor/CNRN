import torch
import numpy as np
from torch.utils.data import Dataset


class CausalDataset(Dataset):
    """
    Dataset for causal models with clearly separated covariates (X), treatment (t), and outcome (y).
    Suitable for use with DragonNet/DragonNRN architectures.
    """

    def __init__(
        self,
        X_covariates: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
        g: np.ndarray = None,
        y_original: np.ndarray = None,
    ):
        """
        Args:
            X_covariates (np.ndarray): Covariate matrix (N, D)
            t (np.ndarray): Binary treatment assignment vector (N,)
            y (np.ndarray): Outcome vector (N,)
            g (np.ndarray, optional): Pre-computed propensity scores (N,)
        """
        assert len(X_covariates) == len(t) == len(y), "All inputs must have the same number of samples"

        self.X = X_covariates.astype(np.float32)
        self.t = t.astype(np.float32).reshape(-1, 1)
        self.y = y.astype(np.float32).reshape(-1, 1)
        self.y_original = None if y_original is None else y_original.astype(np.float32).reshape(-1, 1)
        
        if g is not None:
            self.g = g.astype(np.float32).reshape(-1, 1)
        else:
            self.g = t.astype(np.float32).reshape(-1, 1)
        
        self.sample_idx = np.arange(len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        batch = {
            'features': torch.from_numpy(self.X[idx]),
            'treatment': torch.from_numpy(self.t[idx]),
            'outcome': torch.from_numpy(self.y[idx]),
            'propensity': torch.from_numpy(self.g[idx]),
            'sample_idx': idx
        }
        if self.y_original is not None:
            batch['outcome_original'] = torch.from_numpy(self.y_original[idx])
        return batch
