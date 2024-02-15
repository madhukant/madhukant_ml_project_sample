from Utils import display_dataframe_info
import pandas as pd

class CleanData:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean_the_data(self):

        display_dataframe_info(self.df, 'DataFrame before cleaning')

        self.df = self.df[['datum', 'M01AB','M01AE']]

        self.df['datum'] = pd.to_datetime(self.df['datum'])

        self.df = self.df.melt(id_vars=['datum'], value_vars=['M01AB','M01AE'])

        display_dataframe_info(self.df, 'DataFrame After cleaning')

    def get_cleaned_dataframe(self):
        return self.df
        



