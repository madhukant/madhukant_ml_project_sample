import os
import sys
import warnings

# sys.path.append('../')
# sys.path.append('../src')

sys.path.append(os.path.join(os.getcwd(), 'src'))

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from read.ReadCsvFile import ReadCsvFile
from preprocess.CleanData import CleanData
from preprocess.FeatureExtract import FeatureExtract
from write.WriteCsvFile import WriteCsvFile
from model.RegressionModel import RegressionModel
from evaluate.EvaluateModel import EvaluateModel
from visualize.VisualizeModel import VisualizeModel


class MainProcessing:
    def __init__(self):
        load_dotenv('dev.env')

    def run(self):

        # Read the CSV data
        reader = ReadCsvFile()

        df = reader.read_csv_file(os.getenv('DATA_PATH_ROOT_FOLDER_RAW'))

        # Clean the Data for getting only needed info
        cleaner = CleanData(df)
        cleaner.clean_the_data()

        cleaned_df = cleaner.get_cleaned_dataframe()


        # Do Feature Extraction on the cleaned data
        feature_extractor = FeatureExtract(cleaned_df)
        feature_extractor.do_feature_engineering()
        df_new = feature_extractor.get_latest_df()

        # Save the cleaned data to File
        writer = WriteCsvFile()
        writer.write_to_csv_file(df_new, os.getenv('DATA_PATH_ROOT_FOLDER_PREPROCESSED'))

        # Start Traning the model and saving that to file
        modeler = RegressionModel()
        
        modeler.prepare_data(df_new, target_column='target')
        modeler.print_train_test_data_info()
        modeler.train()
        modeler.save_model(os.getenv('TRAINED_MODEL_PATH'))


        # Get prediction and test data
        pred = modeler.predict()

        X_test, y_test = modeler.get_test_data()

        print(pred)
        print(X_test)
        print(y_test)


        # Start Evaluating the model by getting Mean Square Error, Mean Absolute Error and R squared Scores
        evaluator = EvaluateModel()

        evaluator.evaluate_model(pred, y_test)

        mse, mae, r2 = evaluator.get_scores()

        print('Mean Square Error   =>', mse)
        print('Mean Absolute Error =>', mae)
        print('R squared           =>', r2)


        # Now Visualize the model graph and save graph image to the file - if want to show the graph then pass show_window=True
        visualizor = VisualizeModel()

        visualizor.visualize_model(y_test, pred, show_window=False)


if __name__ == '__main__':

    main = MainProcessing()

    main.run()