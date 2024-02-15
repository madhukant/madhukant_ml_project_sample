import os
import sys
import warnings

sys.path.append(os.path.join(os.getcwd(), 'src'))

import unittest
from dotenv import load_dotenv
from read.ReadCsvFile import ReadCsvFile
from preprocess.CleanData import CleanData
from preprocess.FeatureExtract import FeatureExtract
from write.WriteCsvFile import WriteCsvFile

class TestWriteFile(unittest.TestCase):

    def setUp(self):
        load_dotenv('dev.env')
        warnings.filterwarnings("ignore")

        # Delete the file before running so that after wrtting we could test if file being created or not.
        try:
            os.remove(os.getenv('DATA_PATH_ROOT_FOLDER_PREPROCESSED'))
        except Exception as e:
            print(f"Error While Deleting the file - {os.getenv('DATA_PATH_ROOT_FOLDER_PREPROCESSED')} => {e}")

    def test_clean_data(self):

        reader = ReadCsvFile()

        df = reader.read_csv_file(os.getenv('DATA_PATH_ROOT_FOLDER_RAW'))

        cleaner = CleanData(df)
        cleaner.clean_the_data()

        cleaned_df = cleaner.get_cleaned_dataframe()

        feature_extractor = FeatureExtract(cleaned_df)
        feature_extractor.do_feature_engineering()
        df_new = feature_extractor.get_latest_df()

        writer = WriteCsvFile()
        writer.write_to_csv_file(df_new, os.getenv('DATA_PATH_ROOT_FOLDER_PREPROCESSED'))

        file_exists = os.path.isfile(os.getenv('DATA_PATH_ROOT_FOLDER_PREPROCESSED'))

        self.assertEqual(file_exists, True) # file should exist

        