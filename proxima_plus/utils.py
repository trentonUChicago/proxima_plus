import numpy as np
import random

# For processing data
from sklearn.metrics import pairwise
from dscribe.descriptors import SOAP
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional
from sklearn.pipeline import Pipeline

def make_data_pipeline(soap_kwargs=None, kernel_kwargs=None):
    """Creates data pipeline to be used in ensemble"""
    soap_kwargs   = soap_kwargs   or {}
    kernel_kwargs = kernel_kwargs or {}
    return Pipeline([
        ("soap",   SOAPConverter(**soap_kwargs)),
        ("kernel", ScalableKernel(**kernel_kwargs))
    ])


def radius_of_gyration(atoms):
    cm = atoms.get_center_of_mass()
    disp = np.linalg.norm(atoms.get_positions() - cm, 2, axis=1)
    m = atoms.get_masses()

    return np.dot(m, disp) / np.sum(m)

def get_atom_key(atoms):
    """Gets a hashable key to represent an atom object"""
    pos = np.round(atoms.get_positions(), decimals=6)
    symbols = atoms.get_chemical_symbols()
    return tuple(zip(symbols, map(tuple, pos)))

def scalable_kernel(mol_a: np.ndarray, mol_b: np.ndarray, gamma: float = 1.0) -> float:
    """Compute the scalable kernel between molecules

    Args:
        mol_a (ndarray): Represnetation of a molecule
        mol_b (ndarray): Representation of another molecule
        gamma (float): Kernel parameter
    Returns:
        (float) Similarity between molecules
    """
    return np.exp(-1 * pairwise.pairwise_distances(mol_a, mol_b, 'sqeuclidean', n_jobs=1) / gamma).sum()


class SOAPConverter(BaseEstimator, TransformerMixin):
    """Compute the SOAP descriptors for molecules"""

    def __init__(self, rcut: float = 6, nmax: int = 8, lmax: int = 6, species=frozenset({'C', 'O', 'H', 'N', 'F'})):
        """Initialize the converter
        
        Args:
            rcut (float); Cutoff radius
            nmax (int):
            lmax (int):
            species (Iterable): List of elements to include in potential
        """
        super().__init__()
        self.soap = SOAP(r_cut=rcut, n_max=nmax, l_max=lmax, species=sorted(species))

        self.previous_results = {}

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        result = []
        for x in X:
            atom_key = get_atom_key(x)
            if atom_key in self.previous_results:
                result.append(self.previous_results[atom_key])
            else:
                mat = self.soap.create(x)
                result.append(mat)
                self.previous_results[atom_key] = mat
                
        return result


class ScalableKernel(BaseEstimator, TransformerMixin):
    """Class for computing a scalable atomistic kernel

    This kernel computes the pairwise similarities between each atom in both molecules.
    The total similarity molecules is then computed as the sum of these points
    """

    def __init__(self, max_points: Optional[int] = None, gamma: float = 1.0):
        super(ScalableKernel, self).__init__()
        self.train_points = None
        self.max_points = max_points
        self.gamma = gamma
        # Added by me
        self.id = random.random()

    def fit(self, X, y=None):
        if self.max_points is None:
            # Store the training set
            self.train_points = np.array(X)
        else:
            inds = np.random.choice(len(X), size=(self.max_points))
            self.train_points = np.array(X)[inds]
        self.fitted_ = True
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        K = np.zeros((len(X), len(self.train_points)))
        for i, x in enumerate(X):
            for j, tx in enumerate(self.train_points):
                K[i, j] = scalable_kernel(x, tx, self.gamma)
        return K