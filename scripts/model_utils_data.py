import numpy as np
import pandas as pd

import argparse

RANDOM_STATE = 42

def get_args():
    parser = argparse.ArgumentParser(description="Training pipeline configuration.")
    # Data standardization method
    parser.add_argument(
        "--scale",
        type=str,
        choices=["minmax", "standardization", "none"],
        default="none",
        help="Data scaling method: 'minmax' or 'standardization (Default)'."
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "nn", "rf", "knn", "vc"],
        default="logistic",
        help="Model type to train: 'logistic' (default), 'nn', 'rf', or 'knn'."
    )

    parser.add_argument(
        "--resampler",
        choices=["none", "over-smote", "over-kmeans", "under-random", "under-kmeans", "both", "both-kmeans"],
        type=str,
        default="none",
        help=(
            "Resampling strategy for class imbalance. "
            "'none': no resampling (default); "
            "'over': SMOTE oversampling; "
            "'over-kmeans': SMOTE Kmeans oversampling; "
            "'under-random': random undersampling; "
            "'under-kmeans': KMeans undersampling; "
            "'both': SMOTE oversampling + random undersampling;"
            "'both-kmeans': SMOTE Kmeans oversampling + KMeans undersampling."
        )
    )

    parser.add_argument(
        "--thresholds-num",
        type=int,
        default=0,
        help="Number of threshold values to evaluate when finding the best threshold. Set to 0 to skip threshold search (default: 0)."
    )

    parser.add_argument(
        '--expansion',
        type=int,
        default=0,
        help='Apply polynomial feature expansion. Set to degree (e.g., 2 or 3). Set to 0 to disable.'
    )

    parser.add_argument(
        "--pca",
        nargs='?',
        const=0.95,
        type=float,
        default=0,
        help=(
            "Enable PCA dimensionality reduction. "
            "Use `--pca` to keep 95%% variance by default, or `--pca 10` to reduce to 10 components. "
            "Set to 0 (default) to disable."
        )
    )

    # Flag to enable test mode: train on train+val, evaluate on test (default: False)
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, trains on the combined train and validation sets and evaluates on the test set (default: False)."
    )

    # Flag to apply preprocessing (default: False)
    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply preprocessing steps (default: False)."
    )

    # Flag to save the trained model (default: False)
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save the trained model (default: False)."
    )

    # Flag to use a sample dataset (default: False)
    parser.add_argument(
        "--use-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the sample dataset instead of the full dataset (default: False)."
    )

    # Flag to enable random search for hyperparameter tuning (default: False)
    parser.add_argument(
        "--random-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use random search for hyperparameter tuning (default: False)."
    )

    return parser.parse_args()


def transform_columns(df: pd.DataFrame):
    df = df.copy()
    # to_transform = ['Amount', 'V5', 'V6', 'V7', 'V8', 'V20', 'V21', 'V23', 'V27', 'V28']
    to_transform = ['Time', 'Amount', 'V17', 'V14', 'V12']

    for col in to_transform:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(np.abs(df[col]))

    # df = df.drop(columns=[col for col in to_transform if col in df.columns])
    
    return df


def preprocessing_pipeline(df: pd.DataFrame):
    print("Data is being preprocessed...")

    print("Shape:", df.shape)
    
    print("Removing duplicates...")
    df = df.drop_duplicates()

    print("Transforming columns...")
    df = transform_columns(df)

    print("Final shape:", df.shape)
    print()

    return df


def save_model_and_config(model, config, model_name, config_name):
    """
    Saves the trained model as a .pkl file and its configuration/metrics as a .json file.

    Parameters:
        model: Trained model object
        config (dict): Dictionary of relevant configurations and validation metrics
        output_dir (str): Directory to save the model and config (default: 'saved_model')
    """
    import json
    import pickle
    import os

    CONFIG_PATH = '../config/'
    MODEL_PATH = '../models/'

    # Save model
    model_path = os.path.join(MODEL_PATH, model_name + ".pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save config/metrics
    config_path = os.path.join(CONFIG_PATH, config_name + ".json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Model saved to: {model_path}")
    print(f"Config saved to: {config_path}")


def make_config(args, model_results, threshold_used, training_time, model_file):
    """
    Builds a configuration dictionary that summarizes key model details,
    CLI options, and selected evaluation metrics (rounded to 4 decimal places).

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.
        model_results (dict): Dictionary containing model evaluation metrics.
        threshold_used (float): Threshold used for classification decisions.
        training_time (float): Training duration in seconds.
        model_file (str): Filename of the saved model (.pkl).

    Returns:
        dict: A configuration dictionary with key model info and rounded metrics.
    """

    def round_if_number(val):
        return round(val, 4) if isinstance(val, (float, int)) else val
    
    # Mapping from CLI model choice to full model name
    model_name_map = {
        "logistic": "LogisticRegression",
        "nn": "NeuralNetwork",
        "rf": "RandomForest",
        "knn": "KNearestNeighbors",
        "vc": "VotingClassifier",
    }

    # Map internal resampler argument to readable description
    resampler_map = {
        "none": "No Resampling",
        "over-smote": "Over: SMOTE",
        "over-kmeans": "Over: KMeans-SMOTE",
        "under-random": "Under: Random",
        "under-kmeans": "Under: KMeans",
        "both": "Over: SMOTE + Under: Random",
        "both-kmeans": "Over: KMeans-SMOTE + Under: KMeans"
    }

    config = {
        "model_name": model_name_map.get(args.model, "UnknownModel"), 
        "pkl_name": model_file + ".pkl",
        "random_state": getattr(args, "random_state", RANDOM_STATE),
        "training_time_sec": round_if_number(training_time),
        "evaluated_on": "test" if getattr(args, "test", False) else "validation",
        "search_strategy": "RandomGridSearch" if args.random_search else "None",
        "resampling_strategy": resampler_map.get(args.resampler, "UnknownStrategy"),
        "scaling_method": args.scale,  # 'minmax' or 'standardization'
        "is_sample": args.use_sample,
        "is_PCA": args.pca > 0,
        "is_default_threshold": getattr(args, "thresholds_num", 0) == 0,
        "threshold": round_if_number(threshold_used),
        "poly_expansion_degree": args.expansion,  # 0 for none
    }

    # List of selected metrics to include
    important_metrics = [
        "f1_train", "f1_test", "precision_test", "recall_test", "average_precision_test"
    ]

    for key in important_metrics:
        if key in model_results:
            config[key] = round_if_number(model_results[key])

    return config