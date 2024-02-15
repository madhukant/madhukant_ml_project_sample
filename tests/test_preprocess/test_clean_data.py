import os
import sys
import warnings

sys.path.append(os.path.join(os.getcwd(), 'src'))
warnings.filterwarnings("ignore")

import unittest
from dotenv import load_dotenv
from read.ReadCsvFile import ReadCsvFile
from preprocess.CleanData import CleanData

class TestCleanData(unittest.TestCase):

    def setUp(self):
        load_dotenv('dev.env')
        warnings.filterwarnings("ignore")

    def test_clean_data(self):

        reader = ReadCsvFile()

        df = reader.read_csv_file(os.getenv('DATA_PATH_ROOT_FOLDER_RAW'))

        cleaner = CleanData(df)
        cleaner.clean_the_data()

        cleaned_df = cleaner.get_cleaned_dataframe()

        columns = list(cleaned_df.columns)

        self.assertEqual(len(columns), 3) # there should be three column
        self.assertListEqual(columns, ['datum', 'variable', 'value'])

        