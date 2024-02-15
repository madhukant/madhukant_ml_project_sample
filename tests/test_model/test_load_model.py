import os
import sys
import warnings

sys.path.append(os.path.join(os.getcwd(), 'src'))
warnings.filterwarnings("ignore")

import unittest
from dotenv import load_dotenv
from read.ReadCsvFile import ReadCsvFile
from model.RegressionModel import RegressionModel
from model.LoadTrainedModel import LoadTrainedModel

class TestRegressionModel(unittest.TestCase):

    def setUp(self):
        load_dotenv('dev.env')
        warnings.filterwarnings("ignore")
        self.modeler = None

    def test_read_csv_file(self):
        
        loader = LoadTrainedModel()

        model = loader.load_model(os.getenv('TRAINED_MODEL_PATH'))

        self.assertIsNotNone(model) # model loaded
        