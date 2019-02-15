import numpy as np
import logging
from gp.utils import *


def pick_idxs_greedy(x, y, count, gp_instantiation, epsilon=0):
    """
    Greedily pick to approximately find the set that maximizes the mutual
    information between selected data points and remaining data points.

    [ Algorithm 3, Krause et. al. 2008 ]

    Parameters
    ----------
    x: m x n array
        dataset of features

    y: m-length array
        array of labels

    count: integer
        number of data points to select

    gp_instantiation: zero-parameter function that returns a GaussianProcess


    epsilon: float
        lower bound on absolute value of kernel to be considered

    Returns
    -------
    sel_idxs: list
        indices of data points included, in the order selected
    """
    def entropy_trunc(idxs, idxs_observed):
        idxs_observed = np.array(list(idxs_observed))
        gp = gp_instantiation()
        k = np.abs(gp.kernel(x[idxs,], x[idxs_observed,:]))
        _, trunc = np.nonzero(k > epsilon)
        idxs_observed = idxs_observed[trunc]
        assert len(idxs_observed) > 0
        _ = gp.fit(x[idxs_observed,:], y[idxs_observed])
        _, sigma = gp.predict(x[idxs,:])
        return gaussian_entropy(sigma)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    m, n = x.shape
    all_idxs, sel_idxs = set(np.arange(m)), list()
    deltas, mask = np.zeros(m), np.ones(m)
    neighbors = [None] * m
    kernel = gp_instantiation().kernel

    for i in np.arange(m):
        idxs_observed = np.array(list(all_idxs - {i}))
        k = np.abs(kernel(x[[i],], x[idxs_observed,:]))
        _, trunc = np.nonzero(k > epsilon)
        neighbors[i] = idxs_observed[trunc]

    for i in np.arange(m):
        deltas[i] = entropy_trunc([i], all_idxs - {i})

    for k in range(count):
        # TODO: randomly pick one of max within 0.1 sd of max delta
        logger.info(f"== Max delta: {np.max(deltas * mask)}")
        i = np.random.choice(np.arange(m)) if k == 0 else np.argmax(deltas * mask)
        sel_idxs += [i]
        mask[i] = 0
        for j in filter(lambda l: mask[l], neighbors[i]):
            deltas[j] = entropy_trunc([j], sel_idxs) - \
                        entropy_trunc([j], all_idxs - {j} - set(sel_idxs))

    return sel_idxs

