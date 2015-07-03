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

if __name__ == '__main__':
    unittest.main()
