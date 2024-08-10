import unittest
from pickle import load

import numpy as np

from Packages import Loader, Processor, Predictor

class PipelineTest(unittest.TestCase):

    def setUp(self):
        test_file = './data/train.csv'
        test_target_column = "Survived"
        test_model = './models/clf.pkl'
        test_embarked_encoder = './models/embarked_encoder.pkl'
        test_pclass_encoder = './models/pclass_encoder.pkl'
        test_sex_encoder = './models/sex_encoder.pkl'
        test_scaler = './models/scaler.pkl'

        with open(test_model, 'rb') as f:
            clf = load(f)

        with open(test_embarked_encoder, 'rb') as f:
            embarked_encoder = load(f)

        with open(test_pclass_encoder, 'rb') as f:
            pclass_encoder = load(f)

        with open(test_sex_encoder, 'rb') as f:
            sex_encoder = load(f)

        with open(test_scaler, 'rb') as f:
            scaler = load(f)

        self.__loader = Loader.DataLoader(test_file, None, test_target_column, embarked_encoder.categories_[0], sex_encoder.classes_, pclass_encoder.categories_[0])
        self.__loader.load_data()
        self.__loader.sanity_check()


        self.__preprocessor = Processor.Preprocessor(self.__loader.dataset, embarked_encoder, sex_encoder, pclass_encoder, scaler)
        self.__preprocessor.preprocess_data()

        self.__model = Predictor.ModelManager(self.__preprocessor.dataset, self.__loader.validation_set, clf)
        self.__model.predict()
        self.__model.evaluate()

    def test_required_columns(self):
        required_columns = set(['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Embarked'])
        output_columns = set(self.__loader.dataset.columns.tolist())
        self.assertTrue(required_columns.issubset(output_columns))

    def test_embarked_not_empty(self):
        self.assertFalse(self.__loader.dataset.Embarked.isna().any())

    def test_age_not_empty(self):
        self.assertFalse(self.__preprocessor.preprocessed_data.Age.isna().any())

    def test_sex_column_binarized(self):
        ouptut_values = set(self.__preprocessor.preprocessed_data.Sex.unique().tolist())
        required_values = set([0,1])
        self.assertEqual(ouptut_values, required_values)

    def test_required_processed_columns(self):
        output_columns = set(self.__preprocessor.preprocessed_data.columns.tolist())
        required_columns = set(['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2',
                                'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S'])
        self.assertTrue(required_columns.issubset(output_columns))
        
    def test_processor_output_type(self):
        self.assertTrue(isinstance(self.__preprocessor.dataset, np.ndarray))

    def test_model_ouptut_type(self):
        self.assertTrue(isinstance(self.__model.prediction, np.ndarray))

    def test_model_output(self):
        output_values = set(np.unique(self.__model.prediction).tolist())
        required_columns = set([0,1])
        self.assertEqual(output_values, required_columns)