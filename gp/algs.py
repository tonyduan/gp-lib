import numpy as np
import logging
from gp.utils import *


def pick_idxs_const_gp(x, y, count, gp_instantiation, epsilon_pctle=75, mu=0.1):
    """
    Greedily pick data points to approximately find the set that maximizes the
    mutual information between selected data points and remaining data points.

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
        used to track mean, kernel, noise level

    epsilon_pctle: float between [0, 100]
        in order to speed up computation, when fitting the GP to consider
        selecting data point y we truncate elements x for which
            K(x,y) ≤ ε;
        empirically we calculate ε based on this percentile of kernel scores
        in the entire dataset

    mu: float
        at each step we randomly pick from the set of data points with mutual
        information criterion within the top { max - mu * sd(criterion) }
        (in order to add stochasticity when all points have identica info)

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
    mask = np.ones(m)
    deltas = np.ones(m) * 1e4
    neighbors = [None] * m
    kernel = gp_instantiation().kernel

    epsilon = np.percentile(np.abs(kernel(x, x)), epsilon_pctle)
    logger.info(f"== Greedy sampling epsilon: {epsilon:.3f}")

    for i in np.arange(m):
        idxs_observed = np.array(list(all_idxs - {i}))
        k = np.abs(kernel(x[[i],], x[idxs_observed,:]))
        _, trunc = np.nonzero(k > epsilon)
        neighbors[i] = idxs_observed[trunc]

    median_num_neighbors = np.median([len(n) for n in neighbors])
    logger.info(f"== Greedy sampling size of dataset: {m:.1f}")
    logger.info(f"== Greedy sampling median neighbors: {median_num_neighbors}")

    for i in np.arange(m):
        deltas[i] -= entropy_trunc([i], all_idxs - {i})

    for k in range(count):
        max_val = np.max(deltas * mask)
        idxs_max = deltas * mask > max_val - mu * np.std(deltas * mask)
        i = np.random.choice(np.arange(m)[idxs_max])
        sel_idxs += [i]
        mask[i] = 0
        for j in filter(lambda l: mask[l], neighbors[i]):
            deltas[j] = entropy_trunc([j], sel_idxs) - \
                        entropy_trunc([j], all_idxs - {j} - set(sel_idxs))
        logger.info(f"== Max delta: {max_val:.2f}, Ratio: {sum(idxs_max) / m}")

    return sel_idxs


def pick_idxs_stoch_gp(x, y, h, count, gp_instantiation, epsilon_pctle=75,
                       mu=0.1):
    """
    Greedily pick data points to approximately find the set that maximizes the
    mutual information between selected data points and remaining data points.

    [ Algorithm 3, Krause et. al. 2008 ]

    Parameters
    ----------
    x: m x n array
        dataset of features

    y: m-length array
        array of labels

    h: m x p array
        dataset of features

    count: integer
        number of data points to select

    gp_instantiation: zero-parameter function that returns a GaussianProcess
        used to track mean, kernel, noise level

    epsilon_pctle: float between [0, 100]
        in order to speed up computation, when fitting the GP to consider
        selecting data point y we truncate elements x for which
            K(x,y) ≤ ε;
        empirically we calculate ε based on this percentile of kernel scores
        in the entire dataset

    mu: float
        at each step we randomly pick from the set of data points with mutual
        information criterion within the top { max - mu * sd(criterion) }
        (in order to add stochasticity when all points have identica info)

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
        _ = gp.fit(x[idxs_observed,:], y[idxs_observed], h[idxs_observed,:])
        _, sigma = gp.predict(x[idxs,:], h[idxs,:])
        return gaussian_entropy(sigma)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    m, n = x.shape
    all_idxs, sel_idxs = set(np.arange(m)), list()
    mask = np.ones(m)
    deltas = np.ones(m) * 1e4
    neighbors = [None] * m
    kernel = gp_instantiation().kernel

    epsilon = np.percentile(np.abs(kernel(x, x)), epsilon_pctle)
    logger.info(f"== Greedy sampling epsilon: {epsilon:.3f}")

    for i in np.arange(m):
        idxs_observed = np.array(list(all_idxs - {i}))
        k = np.abs(kernel(x[[i],], x[idxs_observed,:]))
        _, trunc = np.nonzero(k > epsilon)
        neighbors[i] = idxs_observed[trunc]

    median_num_neighbors = np.median([len(n) for n in neighbors])
    logger.info(f"== Greedy sampling size of dataset: {m:.1f}")
    logger.info(f"== Greedy sampling median neighbors: {median_num_neighbors}")

    for i in np.arange(m):
        deltas[i] -= entropy_trunc([i], all_idxs - {i})

    for k in range(count):
        max_val = np.max(deltas * mask)
        idxs_max = deltas * mask > max_val - mu * np.std(deltas * mask)
        i = np.random.choice(np.arange(m)[idxs_max])
        sel_idxs += [i]
        mask[i] = 0
        for j in filter(lambda l: mask[l], neighbors[i]):
            deltas[j] = entropy_trunc([j], sel_idxs) - \
                        entropy_trunc([j], all_idxs - {j} - set(sel_idxs))
        logger.info(f"== Max delta: {max_val:.2f}, Ratio: {sum(idxs_max) / m}")

    return sel_idxs

