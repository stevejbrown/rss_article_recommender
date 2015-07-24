import pandas as pd
import bibtexparser as btp

class BibtexLibrary(object):
    """ Object that reads bibtex libraries and can output cleaned data to train against """

    def __init__(self):
        self.library_df = None

    def read_library(self, library_bib):
        with open(library_bib) as bibtex_file:
            bib_dict = btp.load(bibtex_file)
        self.library_df = pd.DataFrame.from_dict(bib_dict.entries)

    def consolidate(self):
        ''' Reduces the full zotero dataframe to Title and Abstract columns '''
        consolidated_library = self.library_df[['title','abstract']]
        return consolidated_library.rename(columns={'title': 'Title', 'abstract': 'Abstract'})

    def get_library(self):
        return self.consolidate()
