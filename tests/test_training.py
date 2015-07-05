import unittest
import os.path
from article_recommender.load_zotero import ZoteroLibrary
from article_recommender.training import TrainingData

class TestTrainingData(unittest.TestCase):

    def setUp(self):
        test_data = 'tests/test_data/test_zotero_library.csv'
        self.zotero_library = ZoteroLibrary()
        self.zotero_library.read_library(test_data)
        self.test_library = self.zotero_library.get_library()
        self.training_data = TrainingData(self.test_library)

    def test_init(self):
        self.assertIsNotNone(self.training_data.library)

    def test_row_to_words(self):
        words_row_0 = set(['quantifying', 'magnetic', 'nature', 'light', 'emission'])
        words_row_1 = set(('external reflection infrared spectroscopy anisotropic adsorbate '+
                          'layers dielectric substratesmonolayers octadecylsiloxane formed '+
                          'native silicon si sio glass surfaces adsorption dilute solutions '+
                          'oc tadecyltrichlorosilane investigated polarization angle dependent '+
                          'external reflection infrared spectroscopy contrast metal substrates '+
                          'parallel perpendicular vibrational components adsorbate detected '+
                          'dielectric surfaces monolayer reflection spectra show significant '+
                          'changes function light incidence angle polarization infrared radiation '+
                          'contain detailed information surface orientation film molecules '+
                          'spectral simulations based classical electromagnetic theory yield '+
                          'average tilt angle hydrocarbon chains respect surface normal silicon '+
                          'glass surfaces despite apparent structural identity monolayer films '+
                          'silicon glass significant differences observed monolayer reflection '+
                          'spectra resulting purely optical effects substrate').split())

        row_0 = self.training_data.library.iloc[0,:]
        row_1 = self.training_data.library.iloc[1,:]
        self.assertEqual(set(self.training_data.row_to_words(row_0).split()), words_row_0)
        self.assertEqual(set(self.training_data.row_to_words(row_1).split()), words_row_1)

    def test_clean_library(self):
        self.training_data.clean_library()
        self.assertEqual(len(self.training_data.cleaned_library), 243)

    def test_train_vectorizer(self):
        self.training_data.train_vectorizer()
        correct_weights = [7, 6, 6, 104, 33, 10, 6, 8, 12, 7, 12, 11, 6, 9, 8, 9, 8, 6, 10, 8, 6, 6,
                           5, 11, 5, 9, 48, 12, 8, 7, 6, 14, 5, 21, 7, 40, 7, 41, 9, 30, 31, 16, 11,
                           30, 16, 13, 9, 5, 7, 9, 9, 14, 9, 5, 6, 6, 6, 6, 14, 13, 8, 30, 8, 8, 70,
                           5, 11, 5, 5, 6, 6, 5, 5, 6, 6, 24, 13, 5, 14, 13, 10, 9, 12, 30, 5, 6,
                           15, 5, 13, 26, 22, 11, 7, 5, 26, 62, 7, 18, 6, 9, 27, 32, 5, 14, 21, 9,
                           10, 12, 9, 9, 150, 7, 11, 22, 8, 6, 7, 5, 8, 28, 16, 12, 9, 11, 9, 11, 8,
                           5, 15, 8, 5, 20, 6, 16, 5, 5, 5, 5, 5, 7, 5, 10, 8, 7, 9, 7, 6, 6, 5, 63,
                           5, 7, 12, 7, 14, 10, 8, 5, 5, 9, 10, 5, 28, 12, 8, 9, 32, 5, 8, 6, 14, 6,
                           6, 12, 10, 11, 30, 11, 31, 27, 11, 28, 17, 23, 14, 32, 14, 6, 11, 8, 13,
                           31, 7, 16, 7, 27, 18, 17, 10, 9, 9, 6, 22, 6, 13, 6, 9, 12, 17, 5, 5, 16,
                           6, 42, 65, 5, 7, 31, 8, 19, 7, 6, 64, 17, 21, 10, 10, 36, 5, 6, 10, 13,
                           6, 11, 15, 6, 7, 19, 9, 19, 22, 11, 6, 5, 8, 9, 8, 15, 14, 40, 6, 7, 5,
                           26, 5, 8, 5, 24, 24, 5, 45, 13, 22, 14, 68, 16, 6, 8, 11, 17, 5, 6, 11,
                           96, 78, 14, 19, 7, 15, 5, 43, 9, 19, 7, 7, 6, 6, 111, 5, 11, 13, 5, 6, 5,
                           5, 6, 6, 5, 5, 21, 12, 5, 7, 8, 8, 33, 22, 11, 5, 104, 12, 63, 13, 6, 6,
                           5, 26, 13, 16, 5, 8, 5, 11, 5, 6, 9, 11, 17, 6, 8, 11, 16, 5, 10, 5, 86,
                           24, 5, 83, 136, 9, 27, 7, 20, 28, 10, 7, 23, 8, 10, 10, 8, 28, 28, 12, 6,
                           23, 7, 15, 6, 7, 11, 5, 7, 29, 8, 14, 9, 11, 18, 13, 5, 6, 22, 5, 13, 6,
                           11, 5, 8, 31, 5, 9, 6, 6, 8, 12, 7, 7, 6, 10, 16, 5, 5, 5, 18, 6, 15,
                           101, 20, 5, 28, 28, 9, 7, 10, 28, 24, 10, 6, 7, 5, 13, 12, 12, 6, 9, 26,
                           12, 12, 9, 6, 17, 18, 13, 10, 10, 5, 15, 13, 7, 6, 5, 12, 22, 11, 27, 15,
                           7, 6, 14, 6, 5, 17, 10, 18, 19, 5, 16, 11, 5, 16, 11, 5, 6, 7, 13, 6, 6,
                           6, 10, 7, 6, 7, 10, 5, 5, 13, 14, 7, 11, 5, 5, 23, 6, 16, 6, 8, 32, 9,
                           26, 8, 14, 7, 16, 12, 14, 13, 7, 105, 11, 8, 8, 10, 7, 12, 10, 7, 29, 9,
                           8, 41, 19, 10, 5, 7, 7, 12, 50, 15, 6, 7, 5, 6, 12, 6, 8, 33, 86, 5, 12,
                           8, 10, 10, 7, 6, 19, 8, 47, 7, 11, 5, 6, 5, 11, 5, 8, 5, 5, 35, 12, 6,
                           10, 8, 6, 27, 20, 6, 12, 10, 5, 5, 6, 16, 42, 11, 40, 5, 20, 7, 116, 32,
                           64, 6, 7, 12, 10, 6, 7, 5, 21, 6, 13, 14, 7, 11, 5, 9, 5, 6, 16, 5, 16,
                           15, 39, 23, 6, 9, 30, 23, 17, 23, 15, 7, 10, 8, 6, 11, 13, 9, 24, 14, 46,
                           7, 25, 10, 10, 33, 6, 7, 32, 7, 14, 137, 6, 13, 51, 19, 12, 196, 82, 13,
                           36, 6, 6, 6, 8, 13, 6, 13, 7, 19, 15, 9, 9, 15, 7, 6, 7, 9, 6, 6, 10, 10,
                           7, 31, 10, 16, 7, 31, 8, 9, 7, 6, 8, 7, 9, 5, 10, 11, 5, 26, 18, 7, 10,
                           5, 5, 47, 9, 32, 6, 19, 16, 12, 6, 7, 48, 5, 15, 5, 8, 6, 6, 7, 16, 9,
                           23, 30, 12, 6, 10, 61, 7, 6, 136, 12, 7, 70, 8, 7, 8, 15, 12, 6, 23, 10,
                           5, 6, 15, 20, 10, 7, 16, 5, 5, 5, 39, 33, 16, 6, 5, 10, 9, 11, 68, 5, 6,
                           14, 6, 8, 10, 5, 5, 8, 7, 5, 5, 6, 11, 72, 8, 6, 12, 6, 21, 5, 38, 5, 21,
                           6, 8, 28, 5, 6, 5, 18, 13, 36, 6, 7, 10, 21, 13, 5, 7, 5, 16, 7, 9, 5,
                           15, 10, 30, 5, 13, 15, 10, 7, 8, 24, 12, 5, 6, 9, 14, 7, 17, 9, 65, 12,
                           8, 21, 6, 5, 24, 19, 5, 8, 5, 6, 15, 5, 18, 6, 5, 8, 7, 8, 6, 16, 12, 26,
                           32, 27, 16, 6, 32, 9, 23, 14, 6, 11, 7, 62, 14, 9, 6, 9, 8, 14, 6, 7, 20,
                           7, 14, 12, 10, 8, 5, 6, 75, 43, 7, 8, 15, 90, 19, 6, 20, 9, 7, 5, 7, 8,
                           12, 8, 45, 34, 23, 9, 55, 21, 93, 15, 6, 5, 8, 5, 7, 11, 67, 53, 5, 8, 5,
                           6, 16, 19, 31, 63, 24, 20, 21, 38, 7, 5, 29, 24, 5, 5, 6, 74, 12, 5, 8,
                           21, 32, 6, 5, 5, 5, 5, 7, 19, 11, 9, 39, 7, 7, 8, 16, 9, 5, 21, 22, 7,
                           11, 14, 8, 13, 94, 11, 29, 6, 6, 5, 43, 13, 10, 9, 5, 10, 6, 9, 5, 43, 5,
                           9, 13, 27, 24, 16, 21, 5, 55, 7, 6, 57, 78, 13, 9, 6, 5, 5, 17, 9, 13,
                           11, 10, 9, 16, 6, 20, 43, 66, 9, 12, 6, 6, 7, 6, 11, 5, 17, 6, 18, 15, 7,
                           12, 14, 8, 16, 21, 7, 11, 14, 9, 8, 33, 10, 12, 26, 6, 7, 7, 9, 6, 6, 6,
                           10]
        # Check element-wise equality
        all_elem_equal = (self.training_data.weights == correct_weights).all()
        self.assertTrue(all_elem_equal)

if __name__ == '__main__':
    unittest.main()
