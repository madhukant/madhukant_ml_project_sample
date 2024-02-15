from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib

class LoadTrainedModel:

    def __init__(self):
        self.model = None

    def load_model(self, file_path):
        return joblib.load(file_path)
