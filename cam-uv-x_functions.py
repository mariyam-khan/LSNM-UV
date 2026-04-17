import random
import lingam
from torch import nn
import torch
from typing import List, Any, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.utils import check_random_state


def permute_graph(G: np.ndarray, rng:np.random.RandomState):
    d = G.shape[0]
    perm = np.arange(d)
    rng.shuffle(perm)
    G = G[perm,:]
    G = G[:,perm]
    return G



def gen_BA_graph(d: int, degree: int, rng: Any = None):
    assert degree < d
    G = nx.generators.barabasi_albert_graph(n=d, m=degree, seed=rng)
    G = nx.to_numpy_array(G, dtype=int)
    return np.tril(G, k=-1)


def gen_noise(n: int, rng: np.random.RandomState,Gaussian_noise :bool = False):
    scale = rng.uniform(0.5, 2.0)
    if Gaussian_noise:
        e = rng.normal(loc=0, scale=scale, size=n)
    else:
        nt = rng.randint(0, 4)

        if nt == 0: e = rng.uniform(-scale, scale, size=n)
        if nt == 1: e = rng.laplace(loc=0, scale=scale, size=n)
        if nt == 2: e = rng.gumbel(loc=0, scale=scale, size=n)
        if nt == 3: e = rng.exponential(scale=scale, size=n) - scale
    mean = rng.uniform(-1, 1)
    return e + mean


def random_nlfunc(x: np.ndarray, rng: np.random.RandomState):
    # (x + c)^2
    c = rng.uniform(-1, 1)
    z = (x + c) ** 2
    # add constant
    z += rng.uniform(-1, 1)
    # multiple -1 or not
    return z if rng.random() < 0.5 else -z


def gen_adjacency_matrix(
        d: int = 5,
        degree: int = 2,
        graph_type: str = "Gnm",
        permute: bool = True,
        rng: int | np.random.RandomState | None = None):
    rng = check_random_state(rng)

    if graph_type == "BA": G = gen_BA_graph(d, degree, rng)
    else:
        raise ValueError(
            f'No such graph_type: {graph_type}. \
            graph_type list: ["BA"]'
        )
    if permute:
        G = permute_graph(G, rng)
    return G


def to_DataFrame(d, X):
    if d < 10: X = pd.DataFrame(X, columns=[f'x{i}' for i in range(d)])
    elif d < 100: X = pd.DataFrame(X, columns=[f'x{i:01}' for i in range(d)])
    elif d < 1000: X = pd.DataFrame(X, columns=[f'x{i:02}' for i in range(d)])
    elif d < 10000: X = pd.DataFrame(X, columns=[f'x{i:03}' for i in range(d)])
    return X


def gen_data_matrix(
        n: int = 1000,
        G: np.ndarray = gen_adjacency_matrix(),
        Gaussian_noise: bool = False,
        rng: int | np.random.RandomState | None = None):
    rng = check_random_state(rng)
    d = G.shape[0]
    X = np.zeros((n,d))
    for i in range(d): X[:,i] = gen_noise(n,rng,Gaussian_noise)
    G_nx = nx.from_numpy_array(G.T, create_using=nx.DiGraph)
    topological_order = nx.lexicographical_topological_sort(G_nx)
    for i in topological_order:
        predecessors = np.where(G[i,:] > 0)[0]
        for j in predecessors: X[:,i] += random_nlfunc(X[:,j], rng)
        X[:,i] /= np.std(X[:,i])
    return to_DataFrame(d, X)


def gen_dataset(
        n: int = 1000,
        G: np.ndarray = None,
        d: int = 5,
        degree: int = 2,
        graph_type: str = "BA",
        permute: bool = True,
        Gaussian_noise = False,
        rng: int | np.random.RandomState | None = None):
    rng = check_random_state(rng)
    if G is None:
        G = gen_adjacency_matrix(d, degree, graph_type, permute, rng)
    X = gen_data_matrix(n, G, Gaussian_noise = Gaussian_noise,rng = rng)
    return G, X


