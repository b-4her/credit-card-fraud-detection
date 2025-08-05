import pickle
import argparse
import os
import pandas as pd

from model_utils_data import preprocessing_pipeline
from model_utils_eval import evaluate_model

from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

DEFAULT_TEST_FILE = '../data/split/test.csv'
DEFAULT_MODEL_DIR = '../models/'
DEFAULT_MODEL_NAME = 'best_nn_no_search.pkl'

def main():
    parser = argparse.ArgumentParser(description="Load and evaluate a saved model.")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_NAME, help='Name of the saved model file (without path).')
    args = parser.parse_args()

    model_path = os.path.join(DEFAULT_MODEL_DIR, args.model)

    if not os.path.exists(model_path):
        print(f"\nModel file not found: {model_path}")
        return

    print(f"\nLoading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_pkl = pickle.load(f)

    print(f"\nModel loaded successfully: \n{model_pkl}\n")

    model_pipeline = model_pkl['model_pipeline']
    threshold = model_pkl['threshold']

    X_test, y_test = get_test()

    evaluate_model(model_pipeline, threshold, X_test, y_test)

def evaluate_model(model, threshold, X_val, y_val):
    # Predicted probabilities for class 1
    y_val_scores = model.predict_proba(X_val)[:, 1]

    # Apply threshold to get binary predictions
    y_val_pred = (y_val_scores >= threshold).astype(int)

    # Classification metrics (validation)
    f1_val = f1_score(y_val, y_val_pred)
    precision_val = precision_score(y_val, y_val_pred)
    recall_val = recall_score(y_val, y_val_pred)
    ap_val = average_precision_score(y_val, y_val_scores)

    # Print results
    print(f"\nThreshold: {threshold:.4f}")

    print("\n=== Validation Metrics ===")
    print(f"F1 Score:             {f1_val:.4f}")
    print(f"Precision:            {precision_val:.4f}")
    print(f"Recall:               {recall_val:.4f}")
    print(f"Average Precision:    {ap_val:.4f}\n")


def get_test():
    test_df = pd.read_csv(DEFAULT_TEST_FILE)

    test_df = preprocessing_pipeline(test_df)

    X_test = test_df.drop(columns='Class')
    y_test = test_df['Class']

    return X_test, y_test


if __name__ == "__main__":
    main()