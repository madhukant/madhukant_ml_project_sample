import os
import sys
import warnings

sys.path.append(os.path.join(os.getcwd(), 'src'))
warnings.filterwarnings("ignore")

import pickle
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

    def test_regression_model(self):
        reader = ReadCsvFile()

        df = reader.read_csv_file(os.getenv('DATA_PATH_ROOT_FOLDER_PREPROCESSED'))

        modeler = RegressionModel()
        
        modeler.prepare_data(df, target_column='target')
        modeler.print_train_test_data_info()
        modeler.train()
        modeler.save_model(os.getenv('TRAINED_MODEL_PATH'))
        
        file_exists = os.path.isfile(os.getenv('TRAINED_MODEL_PATH'))

        self.assertEqual(file_exists, True) # model should be saved

        pred = modeler.predict()

        self.assertGreaterEqual(len(pred), 1) # check if we are able to get the prediction

        X_test, y_test = modeler.get_test_data()

        self.assertGreaterEqual(len(X_test), 1) # check if we are able to get the X_test
        self.assertGreaterEqual(len(y_test), 1) # check if we are able to get the y_test


        # save few data to pickle file for using it in evaluate and visualize

        data_to_dump = {
            'y_test' : y_test,
            'pred' : pred
        }

        with open(os.getenv('PATH_TO_DUMP_PICKLE_FILE'), 'wb') as f:
            pickle.dump(data_to_dump, f, protocol=pickle.HIGHEST_PROTOCOL)



        