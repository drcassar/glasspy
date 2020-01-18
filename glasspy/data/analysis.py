import numpy as np
from scipy.spatial.distance import cdist


def relativeNeighborhoodDeviation(X,
                                  Y,
                                  distance_threshold,
                                  metric='euclidean'):
    '''Computes the Relative Neighbourhood Deviation (RND).

    RND is used to check the intrinsic deviation in the data. An example of it
    being used can be seen in Figure 3 from Ref. [1].

    Parameters
    ----------
    X : n-d array
        Values of the features (or independent variable). This function uses a
        lot of RAM depending on the size of X.

    Y : 1-d array
        Values of the target (or dependent variable).

    distance_threshold : float
        Minimum distance for two examples from X to be considered part of the
        same neighbourhood. A value of 1% was used in Ref. [1].

    metric : string or callable, optional
        The distance metric to use. See Ref. [2] for more information. If a
        string, the distance function can be ‘braycurtis’, ‘canberra’,
        ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’,
        ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’,
        ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
        ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
        The default value is 'euclidean'. The value 'euclidean' was used in
        Ref. [1].

    Returns
    -------
    relative_neighborhood_error : numpy array
        Array of all the relative neighborhood errors.

    References
    ----------
    [1] Cassar, Daniel R., André C. P. L. F. de Carvalho, and Edgar D. Zanotto.
        “Predicting Glass Transition Temperatures Using Neural Networks.” Acta
        Materialia 159 (October 15, 2018): 249–56.

    [2] SciPyReference Guide
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

    '''
    all_distances = cdist(X, X, metric=metric)
    relative_neighborhood_error = []

    for distance in all_distances:
        logic = distance <= distance_threshold
        y = Y[logic]
        ymax = max(y)
        ymin = min(y)
        relative_error = (ymax - ymin) / (ymax + ymin) * 100
        relative_neighborhood_error.append(relative_error)

    relative_neighborhood_error = np.array(relative_neighborhood_error)

    return relative_neighborhood_error


RND = relativeNeighborhoodDeviation
