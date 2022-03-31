from typing import List

import toolz
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from algorithms.algorithm_typing import Archive, BatchFitnessMeasure


@toolz.curry
def archive_to_knn_novelty(k: int, archive: Archive[np.array]) -> BatchFitnessMeasure:
    def novelty_measure(batch: List[np.array]) -> List[float]:
        combined = archive.retrieve() + batch
        combined = np.row_stack(combined)

        combined_standardized = StandardScaler().fit_transform(combined)

        # each element in the batch is stored in the knn,
        # we add 1 to retrieve it and ignore it (it is 0 distance)
        knn = NearestNeighbors(n_neighbors=k+1)
        knn.fit(combined_standardized)

        distances, indices = knn.kneighbors(batch)
        avg = np.sum(distances, axis=1) / k

        return avg

    return novelty_measure

