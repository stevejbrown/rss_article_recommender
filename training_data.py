import pandas as pd

class TrainingData(object):
    """ TrainingData loads an existing library of articles to train against. """

    def __init__(self, library):
        self.library = library
