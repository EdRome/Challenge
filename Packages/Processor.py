import pandas as pd
import numpy as np
from scipy.stats import norm

class Preprocessor:

    def __init__(self, dataset, embarked_encoder, sex_encoder, pclass_encoder, scaler):
        """Class to preprocess the loaded dataset

        Args:
            embarked_encoder (_type_): Encoder to handle embarked categories
            sex_encoder (_type_): Encoder to binarize sex column
            pclass_encoder (_type_): Encoder to handle PClass categories
            scaler (_type_): Scaler object for data preparation
        """
        self.__dataset = dataset
        self.__preprocessed_data = None
        self.__embarked_encoder = embarked_encoder
        self.__sex_encoder = sex_encoder
        self.__pclass_encoder = pclass_encoder
        self.__scaler = scaler

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
    def embarked_encoder(self):
        """
        Gets the loaded embarked encoder as a sklearn object
        """
        return self.__embarked_encoder
    
    @embarked_encoder.setter
    def embarker_encoder(self, encoder):
        """
        Sets a new embarked encoder as a sklearn object

        Args:
            encoder (sklearn.preprocessing._encoder): The encoder to be loaded
        """
        self.__embarked_encoder = encoder
    
    @property
    def sex_encoder(self):
        """
        Gets the loaded sex encoder as a sklearn object
        """
        return self.__sex_encoder
    
    @sex_encoder.setter
    def sex_encoder(self, encoder):
        """
        Sets a new sex encoder as a sklearn object

        Args:
            encoder (sklearn.preprocessing._encoder): The encoder to be loaded
        """
        self.__sex_encoder = encoder
    
    @property
    def pclass_encoder(self):
        """
        Gets the loaded pclass encoder as a sklearn object
        """
        return self.__pclass_encoder
    
    @pclass_encoder.setter
    def pclass_encoder(self, encoder):
        """
        Sets a new pclass encoder as a sklearn object

        Args:
            encoder (sklearn.preprocessing._encoder): The encoder to be loaded
        """
        self.__pclass_encoder = encoder
    
    @property
    def scaler(self):
        """
        Gets the loaded scaler as a sklearn object
        """
        return self.__scaler
    
    @scaler.setter
    def scaler(self, scaler):
        """
        Sets a new scaler as a sklearn object

        Args:
            scaler (sklearn.preprocessing._data): The scaler to be loaded
        """
        self.__scaler = scaler

    @property
    def preprocessed_data(self):
        return self.__preprocessed_data

    def preprocess_data(self):
        """
        Performs various data preprocessing steps on the loaded dataset.

        This includes:
          - Imputing missing values in the "Embarked" column.
          - Stratified mean imputation for the "Age" column based on "Sex", "Pclass", and "Embarked".
          - Capping outliers in the "Fare" column using Gamma distribution.
          - Binarizing the "Sex" column using the provided encoder.
          - Encoding categorical features ("Embarked" and "Pclass") using the provided encoders.
          - Removing unnecessary columns like "Name", "Ticket", and "Cabin".
          - Selecting only relevant features for the scaler.
          - Applying feature scaling using the provided scaler.

        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        self.impute_age()
        self.cap_outliers_Fare()
        self.binarize_sex_column()
        self.encode_embarked_column()
        self.encode_pclass_column()
        self.__preprocessed_data = self.clean_columns()
        self.scale_dataset()
        
        return self.__dataset
    
    def impute_age(self):
        """
        Imputes age using a stratified strategy based on Sex, Pclass and Embarked columns

        Returns:
            pd.DataFrame: The processed dataset with imputed values.
        """
        group_columns = ['Sex','Pclass','Embarked']
        
        for group_name, group_data in self.__dataset.groupby(group_columns):
            mean_value = group_data['Age'].mean()
            self.__dataset.loc[
                ((self.__dataset[group_columns] == group_name).all(axis=1) & self.__dataset['Age'].isna()), 
                'Age'] = mean_value

        return self.__dataset
    
    def cap_outliers_Fare(self, lower_percentile=0.01, upper_percentile=0.99, k=1, theta=2, epsilon=1e-6):
        """
        Caps outliers of Fare column.
            Args:
                lower_percentile (float): Lower percentile for outlier detection.
                upper_percentile (float): Upper percentile for outlier detection.
                k (int): Shape parameter of the gamma distribution.
                theta (int): Scale parameter of the gamma distribution.
                epsilon (float): Small value to handle values near 0 or 0

          Returns:
                pd.DataFrame: The processed dataset with the capped values.
        """
        # Fill empty values with mean
        self.__dataset['Fare'].fillna(self.__dataset['Fare'].mean(), inplace=True)

        # Add a very small value to handle values near 0 or 0
        self.__dataset['Fare'] += epsilon

        # Transform the data using a log
        log_data = np.log(self.__dataset['Fare'])

        q1 = np.quantile(log_data, 0.25)
        q3 = np.quantile(log_data, 0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr

        # Cap outliers
        capped_log_data = np.clip(log_data, lower_bound, upper_bound)

        # Inverse transformation
        capped_data = np.exp(capped_log_data)
        self.__dataset['Fare'] = capped_data
        
        return self.__dataset
    
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
    
    def binarize_sex_column(self):
        """
        Binarize sex column using the sex_encoder. It returns a binary column.

          Returns:
                pd.DataFrame: The processed dataset with the binarized column
        """
        self.__dataset['Sex'] = self.__sex_encoder.transform(self.__dataset['Sex'])
        return self.__dataset
        
    def clean_columns(self):
        """
        Keep columns necessary for the prediction

          Returns:
                pd.DataFrame: The processed dataset without unnecessary columns
        """
        self.__dataset = self.__dataset[self.__scaler.feature_names_in_]
        return self.__dataset
    
    def encode_embarked_column(self):
        """
        Encode the embarked column using the embarked_encoder. Encoder converts it into a OneHot feature.

          Returns:
                pd.DataFrame: The processed dataset with encoded column, this function doesn't remove the Embarked column
        """
        # Encode the Embarked column
        embarked_encoded = self.__embarked_encoder.transform(self.__dataset[['Embarked']])
        
        columns = [self.__embarked_encoder.feature_names_in_[0] + '_' + category for category in self.__embarked_encoder.categories_[0]]

        # Create a dataframe based on the encoded values
        embarked_encoded_df = pd.DataFrame(data=embarked_encoded, columns=columns)
        
        # Concat the dataset and the dataframe with the encoded values
        self.__dataset = pd.concat(
            [self.__dataset, embarked_encoded_df],
            axis=1
        )
        return self.__dataset
    
    def encode_pclass_column(self):
        """
        Encode the pclass column using the pclass_encoder. Encoder converts it into a OneHot feature.

          Returns:
                pd.DataFrame: The processed dataset with encoded column, this function doesn't remove the Pclass column
        """
        # Encode the Pclass column
        pclass_encoded = self.__pclass_encoder.transform(self.__dataset[['Pclass']])
        
        columns = [self.__pclass_encoder.feature_names_in_[0] + '_' + str(category) for category in self.__pclass_encoder.categories_[0]]

        # Create a dataframe based on the encoded values
        pclass_encoded_df = pd.DataFrame(data=pclass_encoded, columns=columns)
        
        # Concat the dataset and the dataframe with the encoded values
        self.__dataset = pd.concat(
            [self.__dataset, pclass_encoded_df],
            axis=1
        )
        
        return self.__dataset
    
    def scale_dataset(self):
        """
        Scales features using the scaler.

          Returns:
                np.array: The scaled dataset.
        """
        self.__dataset = self.__scaler.transform(self.__dataset)
        return self.__dataset