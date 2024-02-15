import pandas as pd

class ReadCsvFile:

    def __init__(self):
        self.df = None
    
    def read_csv_file(self, file_name: str) -> pd.DataFrame:

        self.df = pd.read_csv(file_name)
        return self.df
