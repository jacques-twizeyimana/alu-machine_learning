#!/usr/bin/env python3

"""
This module contain a function that perfoms PCA
on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """
    Function that perfoms PCA on a dataset

    Arguments:
    X: numpy.ndarray: (n, d) the dataset
        n - no. of data points
        d - no. of dimentions in each point
    var: float: the fraction of variance that the PCA
    should capture

    Returns:
    W: numpy.ndarray: (d, k) the matrix that will be used
    to reduce the dimensionality of the dataset
    """

    # Center the data
    # X = X - np.mean(X, axis=0)

    # # Compute the covariance matrix
    # cov = np.cov(X, rowvar=False)

    # # Compute the eigenvalues and eigenvectors
    # eigvals, eigvecs = np.linalg.eig(cov)

    # # Sort the eigenvectors by decreasing eigenvalues
    # idx = np.argsort(eigvals)[::-1]
    # eigvecs = eigvecs[:, idx]

    # # Compute the total variance
    # total_var = np.sum(eigvals)

    # # Compute the number of components to keep
    # k = 0
    # for i in range(len(eigvals)):
    #     if np.sum(eigvals[:i+1]) / total_var >= var:
    #         k = i + 1
    #         break

    # # Select the k eigenvectors
    # W = eigvecs[:, :k]

    # return W
    u, s, v = np.linalg.svd(X)
    ratios = list(x / np.sum(s) for x in s)
    variance = np.cumsum(ratios)
    nd = np.argwhere(variance >= var)[0, 0]
    W = v.T[:, :(nd + 1)]
    return (W)
