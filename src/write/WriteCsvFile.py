import pandas as pd

class WriteCsvFile:

    def __init__(self):
        pass
    
    def write_to_csv_file(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        df.to_csv(file_path, index=False)