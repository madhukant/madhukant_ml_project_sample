from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib

class RegressionModel:

    def __init__(self):
        self.X_values = None
        self.y_values = None
        self.model = None
        self.lable = None
        self.regressor = None

    def prepare_data(self, df: pd.DataFrame, target_column, train_size=0.8):
        self.X_values = df.drop(columns=[target_column])
        self.y_values = df[target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_values, self.y_values, train_size=train_size, random_state=123)

        self.lable = list(df.columns)

    def print_train_test_data_info(self):

        print('*' * 100)

        print('== Size of Training Data ==')
        print('\tSize of X =>', len(self.X_train))
        print('\tSize of y =>', len(self.y_train))

        print('== Size of Test Data ==')
        print('\tSize of X =>', len(self.X_test))
        print('\tSize of y =>', len(self.y_test))

        print('*' * 100)

    def train(self):

        self.regressor = RandomForestRegressor(n_estimators=10, random_state=123, oob_score=True)

        self.regressor.fit(self.X_train, self.y_train)

    def save_model(self, file_path):
        joblib.dump(self.regressor, file_path)

    def predict(self):
        predictions = self.regressor.predict(self.X_test)
        return predictions

    def get_test_data(self):
        return self.X_test, self.y_test
