import matplotlib.pyplot as plt
import os

class VisualizeModel:
    def __init__(self):
        pass

    def visualize_model(self, y_test, y_pred, show_window=False):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted')

        if os.getenv('GRAPH_PATH'):
            plt.savefig(os.getenv('GRAPH_PATH'))

        if show_window:
            plt.show()