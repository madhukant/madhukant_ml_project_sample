import os
import sys
import warnings

sys.path.append(os.path.join(os.getcwd(), 'src'))

import unittest
from dotenv import load_dotenv
from read.ReadCsvFile import ReadCsvFile
from preprocess.CleanData import CleanData
from preprocess.FeatureExtract import FeatureExtract

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        load_dotenv('dev.env')
        warnings.filterwarnings("ignore")

    def test_clean_data(self):

        reader = ReadCsvFile()

        df = reader.read_csv_file(os.getenv('DATA_PATH_ROOT_FOLDER_RAW'))

        cleaner = CleanData(df)
        cleaner.clean_the_data()

        cleaned_df = cleaner.get_cleaned_dataframe()

        feature_extractor = FeatureExtract(cleaned_df)
        feature_extractor.do_feature_engineering()
        df_new = feature_extractor.get_latest_df()

        columns = list(df_new.columns)

        self.assertEqual(len(columns), 3) # there should be three column
        self.assertListEqual(columns, ['7_days', '14_days', 'target'])

        