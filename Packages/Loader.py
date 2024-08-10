import os
import pandas as pd
from scipy.stats import norm

class DataLoader:

    def __init__(self, file_path: str, dataset: pd.DataFrame, target_column: str, embarked_encoder_categories, sex_encoder_categories, pclass_encoder_categories):
        """This class provides the data loading functionality

        Args:
            file_path (str): Path to the file to load
            dataset (pd.DataFrame): Pandas Dataframe in case loading a dataframe is needed
            target_column (str): Name of the column to take from the dataframe to build a validation set

        Raises:
            Exception: When filepath is not valid.
        """

        self.__file_path = None
        self.__dataset = None
        self.__validation_set = None

        self.__target_column = target_column
        self.__embarked_encoder_categories = embarked_encoder_categories
        self.__sex_encoder_categories = sex_encoder_categories
        self.__pclass_encoder_categories = pclass_encoder_categories

        if file_path is not None and file_path != "":
            file_path_exists = os.path.exists(file_path)
        else:
            file_path_exists = False

        if file_path_exists:
            self.__file_path = file_path
        elif file_path_exists and dataset is not None:
            print("Warning: Both file path and dataset were given. Taking file path")
            self.__file_path = file_path
        elif dataset is not None:
            self.__dataset = dataset
            self.validation_set = target_column
        else:
            raise Exception("File path is not valid and dataset is not give. Provide either an existing file path or a pandas DataFrame")
        
    @property
    def file_path(self) -> str:
        """
        Gets the current file path
        """
        return self.__file_path
    
    @file_path.setter
    def file_path(self, file_path: str):
        """
        Sets a new file path if it is valid and loads the data.
        
        This function doesn't make any processing to the data. It only loads it into memory.
        """
        file_path_exists = os.path.exists(file_path)
        if file_path_exists:
            self.__file_path = file_path
            self.load_data()
        else:
            raise Exception("New file path is not valid, please change it before continue")
    
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

    def load_data(self):
        """
        Loads the data from the specified file path as a pandas DataFrame.

        Handles potential FileNotFoundError and returns None on failure.

        Returns:
            pd.DataFrame: The loaded dataset or None on error.
        """
        try:
            if self.__dataset is None or self.__file_path is not None or self.__file_path != "":
                self.__dataset = pd.read_csv(self.__file_path, dtype={"Pclass":"int", })

                if self.__target_column in self.__dataset.columns:
                    print("Info: Target column exists in the current dataset. Creating a validation set")
                    self.__validation_set = self.__dataset[[self.__target_column]]
                else:
                    print("Warning: Target column is not available on current dataset. It won't be possible to validate prediction")
                    self.__validation_set = None
                return self.__dataset
        except Exception as e:
            print("An unexpected exception ocurrs during data loading: ", e)
            return None
        
    def sanity_check(self):
        """
        Performs sanity checks on the loaded dataset to ensure data integrity.

        Checks include:
          - Presence of all required columns.
          - Valid categories in "Pclass", "Sex", and "Embarked" columns (if no missing values).

        Raises:
            Exception: If any sanity check fails.
        """
        required_columns = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                            'Parch', 'Embarked']
        
        # Validate if the required columns is a subset of the dataset's columns
        if not set(required_columns).issubset(set(self.__dataset.columns)):
            raise Exception("Required columns are not in the given dataset.\nRequired columns {}\nGiven columns {}".format(
                required_columns, self.__dataset.columns.tolist()))
        
        # Validate if all Pclass categories exists into the pclass_encoder.
        if not all([value in self.__pclass_encoder_categories for value in self.__dataset['Pclass'].unique()]):
            print("Warning: The dataset has missing classes")

        # Validate if all Sex categories exists into the sex_encoder.
        if not all([value in self.__sex_encoder_categories for value in self.__dataset['Sex'].unique()]):
            raise Exception('Unexpected category on sex column')
        
        # Validate if all Embarked categories exists into the embarked_encoder.
        if not all([value in self.__embarked_encoder_categories for value in self.__dataset['Embarked'].unique()]):
            if self.__dataset['Embarked'].isna().any():
                print("Warning: Embarked column has a missing value. Imputation process will fix it.")
                self.impute_embarked()
            else:
                raise Exception('Unexpected category on embarked column')
            
    def impute_embarked(self):
        """
        Imputes embarked column using the Fare to identify belonging class based on its probability distribution.

          Returns:
                pd.DataFrame: The processed dataset with imputed values.
        """
        # Generate strata
        grouped_data = self.__dataset.groupby("Embarked")

        # Gets the normal distribution for each stratum
        fare_distributions = {name: norm.fit(group['Fare']) for name, group in grouped_data}

        # Get the most likely belonging class for all categories
        self.__dataset['most_likely_class'] = self.__dataset['Fare'].map(
                lambda ts: {embarked: norm.pdf(ts, distribution[0], distribution[1]) for embarked, distribution in fare_distributions.items()}
        ).map(max)

        # Impute the missing values using the most likley class
        self.__dataset['Embarked'].fillna(self.__dataset['most_likely_class'], inplace=True)

        return self.__dataset