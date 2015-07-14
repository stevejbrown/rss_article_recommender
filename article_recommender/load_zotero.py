import pandas as pd

class ZoteroLibrary(object):
    """ Object that reads Zotero libraries and can output cleaned data to train against """

    def __init__(self):
        self.library_df = None

    def read_library(self, library_csv):
        self.library_df = pd.read_csv(library_csv)

    def consolidate(self):
        ''' Reduces the full zotero dataframe to Title and Abstract columns '''
        consolidated_library = self.library_df[['Title','Abstract Note']]
        return consolidated_library.rename(columns={'Abstract Note': 'Abstract'})

    def get_library(self):
        return self.consolidate()
