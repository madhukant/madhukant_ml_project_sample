import os
import sys
import warnings

sys.path.append(os.path.join(os.getcwd(), 'src'))
warnings.filterwarnings("ignore")

import pickle
import unittest
from dotenv import load_dotenv

from visualize.VisualizeModel import VisualizeModel

class TestVisualizeModel(unittest.TestCase):

    def setUp(self):
        load_dotenv('dev.env')
        warnings.filterwarnings("ignore")

    def test_evaluate_model(self):

        with open(os.getenv('PATH_TO_DUMP_PICKLE_FILE'), 'rb') as f:
            loaded_data = pickle.load(f)
        
        pred = loaded_data['pred']
        y_test = loaded_data['y_test']

        visualizor = VisualizeModel()

        visualizor.visualize_model(y_test, pred, show_window=False)
        

        # Visualized graph should be generated
        file_exists = os.path.isfile(os.getenv('GRAPH_PATH'))

        self.assertEqual(file_exists, True) # graph image file should be saved


        