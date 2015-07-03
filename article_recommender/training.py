import pandas as pd
import re
from nltk.corpus import stopwords

class TrainingData(object):
    """ TrainingData loads an existing library of articles to train against. """

    def __init__(self, library):
        ''' library should be a dataframe with 'Title' and 'Abstract' columns '''
        self.library = library

    def row_to_words(self, row):
        ''' Function to convert a raw entry in library entry to a string of words
            Input: A single row from the library
            Output: A single processed string
        '''
        # Merge columns
        text = row.fillna("")
        text = "".join(text)

        # Remove non-letters
        letters_only = re.sub("[^a-zA-z]"," ", text)

        # Convert to lower case, split into individual words
        words = letters_only.lower().split()

        # load stop words (into a set since searching sets is much faster)
        stops = set(stopwords.words("english"))

        # Remove stop words
        meaningful_words = [w for w in words if not w in stops]

        # Join the words back into one string separated by a space,
        # and return the result
        return(" ".join(meaningful_words))
