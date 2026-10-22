#!/usr/bin/env python3

"""
This module contain a function that perfoms PCA
on a dataset"""

import numpy as np


def pca(X, ndim):
    """
    Function that perfoms PCA on a dataset

    X: numpy.ndarray: (n, d) the dataset
        n - no. of data points
        d - no. of dimentions in each point
    ndim: new dimensionality of transformed X

    Returns:
    T: numpy.ndarray: (n, ndim) the transformed
    version of X
    """
    # X = X - np.mean(X, axis=0)
    # cov = np.cov(X, rowvar=False)
    # eigvals, eigvecs = np.linalg.eig(cov)
    # idx = np.argsort(eigvals)[::-1]
    # eigvecs = eigvecs[:, idx]
    # W = eigvecs[:, :ndim]
    # T = np.matmul(X, W)

    # return T
    mean = np.mean(X, axis=0, keepdims=True)
    A = X - mean
    u, s, v = np.linalg.svd(A)
    W = v.T[:, :ndim]
    T = np.matmul(A, W)
    return (T)
