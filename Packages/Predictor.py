import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score

class ModelManager:

    def __init__(self, dataset: np.array, validation_set: pd.DataFrame, classifier):
        """Class to handle model predictions

        Args:
            dataset (np.array): Numpy array containing the data for prediction
            validation_set (pd.DataFrame): Pandas DataFrame to validate the prediction (optional)
            classifier (_type_): Classification model made with sklearn.
        """

        self.__dataset = dataset
        self.__validation_set = validation_set
        self.__classifier = classifier

    @property
    def validation_set(self):
        """
        Gets the loaded dataset, if any
        """
        return self.__validation_set
    
    @validation_set.setter
    def validation_set(self, validation_set):
        if isinstance(validation_set, str) and validation_set in self.__dataset.columns:
            self.__validation_set = self.__dataset[[validation_set]]
        elif isinstance(validation_set, pd.DataFrame) and validation_set.shape[1] == 1 and validation_set.shape[0] == self.__dataset.shape[0]:
            unique_values = set(validation_set.iloc[:,0].unique())
            expected_values = set([0,1])
            
            if expected_values.issubset(unique_values):
                self.__validation_set = validation_set
            else:
                raise Exception("Values on validation set are not binary. Expecting 0,1 values.")
        else:
            raise Exception("Validation set is not valid, if you provided a string, ensure it exists in the current dataset, otherwise change it. If you provided a dataframe, ensure it is a column dataframe and it has 0 and 1 values, and validation set is the same size as the given dataset")
        
    @property
    def dataset(self) -> pd.DataFrame:
        """
        Gets the loaded dataset as a pandas DataFrame.
        """
        return self.__dataset
    
    @dataset.setter
    def dataset(self, df: pd.DataFrame):
        """
        Sets a new dataset as a pandas DataFrame and performs sanity checks.

        Args:
            df (pd.DataFrame): The dataset to be loaded.
        """
        tmp_df = self.__dataset.copy()
        self.__dataset = df
        try:
            self.sanity_check()
        except Exception as e:
            self.__dataset = tmp_df
            print("Error: An error occurrs during sanity check. Unable to change to given dataset")
            print("Error: ", e)

    @property
    def classifier(self):
        """
        Gets the loaded classification model as a sklearn object
        """
        return self.__classifier
    
    @classifier.setter
    def classifier(self, model):
        """
        Sets a new classification model as a sklearn object

        Args:
            scaler (sklearn): The classification model to be loaded
        """
        if not self.__preprocessed_data:
            raise Exception("The data is not preprocessed, hence it is not possible to evaluate new model performance before replacing it.")
        
        if self.__validation_set is None:
            raise Exception("There are no validation set, it's not possible to continue since there is no possiblity to validate new model's performance. Try changing the dataset to one with the target column to continue")
        
        self.predict()
        prev_f1_score = self.evaluate(None, False)
        
        tmp_model = self.__classifier
        self.__classifier = model
        
        self.predict()
        new_f1_score = self.evaluate(None, False)
        
        if prev_f1_score > new_f1_score:
            print("Warning: New model performance is worst than the previous one. Improve it before replacing it.\nCurrent model f1 score: {}\nNew model f1 score: {}".format(prev_f1_score, new_f1_score))
            self.__classifier = tmp_model
        elif new_f1_score > 0.98:
            print("Warning: New model performance is showing a f1 score close to 1. Ensure it is not overfitted before replacing it.\nNew Model f1 score: {}".format(new_f1_score))
            self.__classifier = tmp_model
        else:
            print("New model performance is better, changing to new version.\nNew model f1 score: {}\nPrevious model f1 score: {}".format(new_f1_score, prev_f1_score))
        
    @property
    def prediction(self):
        """
        Gets the prediction made by the current model, if any
        """
        if self.__prediction is None:
            print("Warning: No prediction was made. Running a prediction on the current data")
            if self.__preprocessed_data:
                self.predict()
            else:
                print("Warning: No prediction was possible to be made. Please, run the preprocessing step first before attempting to make a prediction.")
        
        return self.__prediction
    
    def predict(self, dataset=None):
        """
        Predicts labels for the loaded dataset using the trained classifier. If dataset is given, then predict over this new dataset

        Assumes the dataset is already preprocessed and scaled.

        Returns:
            np.ndarray: The predicted labels.
        """
        self.__prediction = self.__classifier.predict(self.__dataset)

        pd.DataFrame(
            data=self.__prediction,
            columns=['Survived']
        ).to_csv(
            './data/prediction.csv', index=False
        )

        return self.__prediction
    
    def evaluate(self, y_true=None, use_classification_report=True):
        """
        Evaluates classification model performance on true labels.
    
        Prints the classification report that contains precision, recall, f1 score for each label
        and an overall evaluation.

        Assumes the input labels are in the same order as the loaded dataset.

        Args:
            y_true (pd.DataFrame): Contains the true labels of the loadad dataset.
            user_classification_report (bool): Defaults true, in False case, then return f1 score.
        """
        if self.__validation_set is not None:
            if not y_true:
                y_true = self.__validation_set
            elif y_true.shape[0] != self.__prediction.shape[0]:
                raise Exception("Validation set and prediction labels are not the same size. Evaluation can be done.")

            if use_classification_report:
                print(classification_report(y_true, self.__prediction))
            else:
                return f1_score(y_true, self.__prediction)
        else:
            print("Warning: No valid set was given.")