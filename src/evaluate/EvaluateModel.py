from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class EvaluateModel:

    def __init__(self):
        self.mse = None
        self.mae = None
        self.r2 = None

    def evaluate_model(self, pred, y_test):
        y_pred = pred

        self.mse = mean_squared_error(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        
    def get_scores(self):
        return self.mse, self.mae, self.r2