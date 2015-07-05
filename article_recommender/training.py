import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

class TrainingData(object):
    """ TrainingData loads an existing library of articles to train against. """

    def __init__(self, library):
        ''' library should be a dataframe with 'Title' and 'Abstract' columns '''
        self.library = library
        self.cleaned_library = None
        self.vectorizer = CountVectorizer(analyzer = "word",
                                          tokenizer = None,
                                          preprocessor = None,
                                          stop_words = None,
                                          max_features = 1000)
        self.weights = None

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

    def clean_library(self):
        ''' Cleans the library with row_to_words and stores it in self.cleaned_library '''
        self.cleaned_library = self.library.apply(self.row_to_words, axis = 1)
        pass

    def train_vectorizer(self):
        ''' Trains the CountVectorizer (bag of words) model on the training data '''
        if type(self.cleaned_library) == type(None):
            self.clean_library()
        train_data_features = self.vectorizer.fit_transform(self.cleaned_library)
        train_data_features = train_data_features.toarray() # Numpy arrays are easier to work with
        self.weights = train_data_features.sum(axis=0)

        # Return a list of (word, count) touples for possible output
        vocab = self.vectorizer.get_feature_names()
        return [(word, count) for word, count in
                sorted(zip(vocab, self.weights), key=lambda tup: tup[1], reverse=True)]
