from Utils import display_dataframe_info
import pandas as pd

class FeatureExtract:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df_preprocessed = None

    def do_feature_engineering(self):

        self.df['7_days'] = self.df.rolling(window=7).mean()

        self.df['14_days'] = self.df[['value']].rolling(window=14).mean()

        self.df = self.df[self.df['variable'] == 'M01AB']

        self.df['target'] = self.df[['value']].shift(-7)

        self.df_preprocessed = self.df[['7_days', '14_days', 'target']]

        self.df_preprocessed = self.df_preprocessed.dropna()

        display_dataframe_info(self.df_preprocessed, 'DataFrame After Feature Extraction')

    def get_latest_df(self):
        return self.df_preprocessed



