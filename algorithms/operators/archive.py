from toolz import random_sample


class ProbabilisticArchive:
    def __init__(self, archive_pr: float):
        self.pr = archive_pr
        self._archive = []

    def store(self, items):
        to_add = list(random_sample(self.pr, items))

        self._archive.extend(to_add)

    def retrieve(self):
        return self._archive.copy()