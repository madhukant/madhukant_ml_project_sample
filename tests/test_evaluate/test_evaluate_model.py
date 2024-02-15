import os
import sys
import warnings

sys.path.append(os.path.join(os.getcwd(), 'src'))
warnings.filterwarnings("ignore")

import pickle
import unittest
from dotenv import load_dotenv

from evaluate.EvaluateModel import EvaluateModel

class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        load_dotenv('dev.env')
        warnings.filterwarnings("ignore")

    def test_evaluate_model(self):

        with open(os.getenv('PATH_TO_DUMP_PICKLE_FILE'), 'rb') as f:
            loaded_data = pickle.load(f)
        
        pred = loaded_data['pred']
        y_test = loaded_data['y_test']

        evaluator = EvaluateModel()

        evaluator.evaluate_model(pred, y_test)

        mse, mae, r2 = evaluator.get_scores()

        self.assertIsNotNone(mse)
        self.assertIsNotNone(mae)
        self.assertIsNotNone(r2)
        