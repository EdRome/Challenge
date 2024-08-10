# Execution
To execute the process, the command line should look like this
> python main.py

It has additional parameters:

FILE_PATH: The file path for the dataset to be used.
TARGET_COLUMN: It is the target column used for validation. The validations are performd only if this column is given and inside the validation dataset
MODEL: Used to change the default model, it should be stored as pickle and the argument recieves the path
EMBARKED_ENCODER: Encoder used to OneHot the embarked column. It should be stored as pickle
PCLASS_ENCODER: Encoder used to OneHot the pclass column. It should be stored as pickle
SEX_ENCODER: Encoder used to binarize the sex column. It should be stored as pickle
SCALER: To scale the dataset. It should be stored as pickle

```
Here is the help detailed.

usage: Rappi Challenge [-h] [-fp FILE_PATH] [-tc TARGET_COLUMN] [-m MODEL] [-ee EMBARKED_ENCODER] [-pc PCLASS_ENCODER]
                       [-se SEX_ENCODER] [-s SCALER]

Predict the Titanic Challenge

options:
  -h, --help            show this help message and exit
  -fp FILE_PATH, --file-path FILE_PATH
                        Path to file to predict. It is recommended to include Survived column to validate predictions.
                        By default uses data/train.csv
  -tc TARGET_COLUMN, --target_column TARGET_COLUMN
                        Target column to use in validation. By default uses Survived
  -m MODEL, --model MODEL
                        Path to new classification model. Must be in pickle format. By default uses models/clf.pkl
  -ee EMBARKED_ENCODER, --embarked-encoder EMBARKED_ENCODER
                        Path to new Embarked encoder. Must be a OneHot Encoder and pickle formated By default uses
                        models/embarked_encoder.pkl
  -pc PCLASS_ENCODER, --pclass-encoder PCLASS_ENCODER
                        Path to new Pclass encoder. Must be a OneHot Encoder. By default uses
                        models/pclass_encoder.pkl
  -se SEX_ENCODER, --sex-encoder SEX_ENCODER
                        Path to new Sex encoder. Must be a binarizer. By default uses models/sex_encoder.pkl
  -s SCALER, --scaler SCALER
                        Path to new scaler. By default uses models/scaler.pkl
```