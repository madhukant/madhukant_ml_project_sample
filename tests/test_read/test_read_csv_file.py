import os
import sys
import warnings

sys.path.append(os.path.join(os.getcwd(), 'src'))
warnings.filterwarnings("ignore")

import unittest
from dotenv import load_dotenv
from read.ReadCsvFile import ReadCsvFile

class TestReadCsvFile(unittest.TestCase):

    def setUp(self):
        load_dotenv('dev.env')
        warnings.filterwarnings("ignore")

    def test_read_csv_file(self):
        reader = ReadCsvFile()

        df = reader.read_csv_file(os.getenv('DATA_PATH_ROOT_FOLDER_RAW'))

        rows, cols = df.shape

        self.assertGreaterEqual(rows * cols, 1) # rows and cols should be more than 1

