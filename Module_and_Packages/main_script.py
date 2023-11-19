# data_science_package/main_script.py
from data.loader import load_data
from data.preprocessing import preprocess_data
from models.classifier import Classifier

def main():
    # Load data
    data = load_data("example_data.csv")

    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Train a classifier
    classifier = Classifier()
    classifier.train(preprocessed_data, labels)

    # Make predictions
    predictions = classifier.predict(new_data)
    print(predictions)

if __name__ == "__main__":
    main()
