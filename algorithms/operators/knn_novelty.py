from functools import lru_cache
from hashlib import sha1

import toolz
import numpy as np

from numpy import all, array, uint8

# copied from https://stackoverflow.com/questions/1939228/constructing-a-python-set-from-a-numpy-matrix/5173201#5173201
from sklearn.neighbors import NearestNeighbors


class hashable(object):
    r'''Hashable wrapper for ndarray objects.

        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.

        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).
    '''

    def __init__(self, wrapped, tight=False):
        r'''Creates a new hashable object encapsulating an ndarray.

            wrapped
                The wrapped ndarray.

            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        '''
        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        r'''Returns the encapsulated ndarray.

            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped


@lru_cache(1)
def _get_knn(n, archive):
    archive = archive.unwrap()

    knn = NearestNeighbors(n_neighbors=n)
    knn.fit(archive)

    return knn


@toolz.curry
def knn_novelty(n, bc, archive):
    knn = _get_knn(n + 1, hashable(archive))

    distances, indices = knn.kneighbors(bc.reshape(1, -1))
    avg = np.sum(distances) / n

    return avg
