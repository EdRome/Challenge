import argparse
from pickle import load
from Packages import Loader, Processor, Predictor

print("Starting...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Rappi Challenge",
        description="Predict the Titanic Challenge"
    )

    parser.add_argument("-fp", "--file-path", default='./data/train.csv', help='Path to file to predict. It is recommended to include Survived column to validate predictions. By default uses data/train.csv')
    parser.add_argument("-tc", "--target_column", default="Survived", help="Target column to use in validation. By default uses Survived", required=False)
    parser.add_argument("-m", "--model", default='./models/clf.pkl', help='Path to new classification model. Must be in pickle format. By default uses models/clf.pkl', required=False)
    parser.add_argument("-ee", "--embarked-encoder", default='./models/embarked_encoder.pkl', help='Path to new Embarked encoder. Must be a OneHot Encoder and pickle formated By default uses models/embarked_encoder.pkl', required=False)
    parser.add_argument("-pc", "--pclass-encoder", default='./models/pclass_encoder.pkl', help='Path to new Pclass encoder. Must be a OneHot Encoder. By default uses models/pclass_encoder.pkl', required=False)
    parser.add_argument("-se", "--sex-encoder", default='./models/sex_encoder.pkl', help='Path to new Sex encoder. Must be a binarizer. By default uses models/sex_encoder.pkl', required=False)
    parser.add_argument("-s", "--scaler", default='./models/scaler.pkl', help='Path to new scaler. By default uses models/scaler.pkl', required=False)

    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        clf = load(f)

    with open(args.embarked_encoder, 'rb') as f:
        embarked_encoder = load(f)

    with open(args.pclass_encoder, 'rb') as f:
        pclass_encoder = load(f)

    with open(args.sex_encoder, 'rb') as f:
        sex_encoder = load(f)

    with open(args.scaler, 'rb') as f:
        scaler = load(f)

    loader = Loader.DataLoader(args.file_path, None, "Survived", embarked_encoder.categories_[0], sex_encoder.classes_, pclass_encoder.categories_[0])
    loader.load_data()
    loader.sanity_check()

    preprocessor = Processor.Preprocessor(loader.dataset, embarked_encoder, sex_encoder, pclass_encoder, scaler)
    preprocessor.preprocess_data()

    model = Predictor.ModelManager(preprocessor.dataset, loader.validation_set, clf)
    model.predict()
    model.evaluate()