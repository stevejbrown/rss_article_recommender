import unittest
import os.path
from article_recommender.load_zotero import ZoteroLibrary

class TestZoteroLibrary(unittest.TestCase):

    def setUp(self):
        self.test_library = ZoteroLibrary()
        self.test_data = 'tests/test_data/test_zotero_library.csv'

    def test_read_library(self):
        self.test_library.read_library(self.test_data)
        self.assertIsNotNone(self.test_library.library_df)

    def test_get_library(self):
        self.test_library.read_library(self.test_data)
        out_library = self.test_library.get_library()
        col_names = ['Title', 'Abstract']
        for name in col_names:
            self.assertTrue(name in list(out_library))

if __name__ == '__main__':
    unittest.main()
