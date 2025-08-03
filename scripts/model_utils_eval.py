import numpy as np
import random

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import (f1_score, recall_score, 
                             precision_score, average_precision_score)

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import VotingClassifier

from sklearn.pipeline import Pipeline as normal_Pipeline
from imblearn.pipeline import Pipeline as imb_Pipeline

from scipy.stats import loguniform, randint

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def evaluate_model(model, threshold, X_train, y_train, X_val, y_val):
    # Predicted probabilities for class 1
    y_train_scores = model.predict_proba(X_train)[:, 1]
    y_val_scores = model.predict_proba(X_val)[:, 1]

    # Apply threshold to get binary predictions
    y_train_pred = (y_train_scores >= threshold).astype(int)
    y_val_pred = (y_val_scores >= threshold).astype(int)

    # Classification metrics (train)
    f1_train = f1_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    ap_train = average_precision_score(y_train, y_train_scores)

    # Classification metrics (validation)
    f1_val = f1_score(y_val, y_val_pred)
    precision_val = precision_score(y_val, y_val_pred)
    recall_val = recall_score(y_val, y_val_pred)
    ap_val = average_precision_score(y_val, y_val_scores)

    # Print results
    print(f"\nThreshold: {threshold:.4f}")

    print("=== Training Metrics ===")
    print(f"F1 Score:             {f1_train:.4f}")
    print(f"Precision:            {precision_train:.4f}")
    print(f"Recall:               {recall_train:.4f}")
    print(f"Average Precision:    {ap_train:.4f}")

    print("\n=== Validation Metrics ===")
    print(f"F1 Score:             {f1_val:.4f}")
    print(f"Precision:            {precision_val:.4f}")
    print(f"Recall:               {recall_val:.4f}")
    print(f"Average Precision:    {ap_val:.4f}\n")

    return {
        "f1_train": f1_train,
        "f1_test": f1_val,
        "precision_train": precision_train,
        "precision_test": precision_val,
        "recall_train": recall_train,
        "recall_test": recall_val,
        "average_precision_train": ap_train,
        "average_precision_test": ap_val,
    }


def search_and_train(X_train, y_train, steps, args):
    '''
    Train and tune one or more classification models using randomized hyperparameter search 
    and cross-validation. Supports logistic regression, random forest, MLP neural network, 
    K-nearest neighbors (KNN), and soft voting classifier (VC) based on top 3 models.

    Parameters:
        X_train : Feature matrix used for training.
        y_train : Binary target labels corresponding to X_train.
        steps : A list of preprocessing pipeline steps (e.g., scaling, sampling).
        args : Parsed command-line arguments containing model type and search settings.

    Returns:
        best_model : The trained model with the highest cross-validated F1 score.
                     If 'vc' is selected, returns a soft voting classifier built from
                     the top 3 models based on F1 score.
    '''

    models = {}
    scores = {}

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Logistic Regression Search
    if args.model in ['logistic', 'vc']:
        pipe_steps = steps.copy()
        pipe_steps.append(('model', LogisticRegression(random_state=RANDOM_STATE)))
        if args.resampler == 'none':
            pipe = normal_Pipeline(pipe_steps)
        else:
            pipe = imb_Pipeline(pipe_steps)

        param_dist = {
            'model__C': loguniform(1e-4, 1e2),
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear', 'saga'],
            'model__max_iter': [3000, 4000, 5000]
        }

        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist,
            n_iter=20, scoring='f1', cv=cv_splitter, n_jobs=-1, random_state=RANDOM_STATE,
            verbose=1
        )
        search.fit(X_train, y_train)

        models['logistic'] = search.best_estimator_
        scores['logistic'] = search.best_score_

    # 2. Random Forest Search
    if args.model in ['rf', 'vc']:
        pipe_steps = steps.copy()
        pipe_steps.append(('model', RandomForestClassifier(random_state=RANDOM_STATE)))
        if args.resampler == 'none':
            pipe = normal_Pipeline(pipe_steps)
        else:
            pipe = imb_Pipeline(pipe_steps)

        param_dist = {
            'model__n_estimators': randint(50, 300),
            'model__max_depth': randint(3, 20),
            'model__min_samples_split': randint(2, 10),
            'model__min_samples_leaf': randint(1, 5)
        }

        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist,
            n_iter=20, scoring='f1', cv=cv_splitter, n_jobs=-1, random_state=RANDOM_STATE,
            verbose=1
        )
        search.fit(X_train, y_train)

        models['rf'] = search.best_estimator_
        scores['rf'] = search.best_score_

    # 3. Neural Network (MLP) Search
    if args.model in ['nn', 'vc']:
        pipe_steps = steps.copy()
        pipe_steps.append(('model', MLPClassifier(max_iter=500)))
        if args.resampler == 'none':
            pipe = normal_Pipeline(pipe_steps)
        else:
            pipe = imb_Pipeline(pipe_steps)

        param_dist = {
            'model__hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': loguniform(1e-5, 1e-1),
            'model__solver': ['adam', 'sgd'],
        }

        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist,
            n_iter=20, scoring='f1', cv=cv_splitter, n_jobs=-1, random_state=RANDOM_STATE,
            verbose=1
        )
        search.fit(X_train, y_train)

        models['nn'] = search.best_estimator_
        scores['nn'] = search.best_score_

    # 4. KNN Search
    if args.model in ['knn', 'vc']:
        pipe_steps = steps.copy()
        pipe_steps.append(('model', KNeighborsClassifier()))
        if args.resampler == 'none':
            pipe = normal_Pipeline(pipe_steps)
        else:
            pipe = imb_Pipeline(pipe_steps)

        param_dist = {
            'model__n_neighbors': randint(3, 30),
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan', 'minkowski']
        }

        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist,
            n_iter=20, scoring='f1', cv=cv_splitter, n_jobs=-1, random_state=RANDOM_STATE,
            verbose=1
        )
        search.fit(X_train, y_train)

        models['knn'] = search.best_estimator_
        scores['knn'] = search.best_score_

    # Output section: voting or best single model
    if args.model == 'vc':
        top_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        estimators = [(name, models[name]) for name, _ in top_models]

        voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1, verbose=True)
        voting.fit(X_train, y_train)

        print("Voting among top models:")
        for name, score in top_models:
            print(f"{name}: F1 CV Score = {score:.4f}")
        return voting
    else:
        best_name, best_score = max(scores.items(), key=lambda x: x[1])
        best_model = models[best_name]
        print(f"\nBest Model: {best_name} (F1 CV Score = {best_score:.4f})")
        return best_model
    

def find_best_threshold(model, X_train, y_train, thresholds_num=100):
    '''
    Find the optimal decision threshold that maximizes the F1 score.

    Parameters:
        model : A trained classification model that supports `predict_proba`
        X_train : Feature matrix used to generate prediction probabilities
        X_target : Ground truth labels (0 or 1)
        thresholds_num : Number of thresholds to evaluate between 0 and 1 (default=100)

    Returns:
        best_threshold : The threshold value that yields the highest F1 score
    '''

    # predicted probabilities (to avoid using the default threshold 0.5)
    y_pred = model.predict_proba(X_train)[:, 1]
    # y_pred contains probs for the element to be in the target class, 0.99 -> 99% it is fraud.

    # Create an array of evenly spaced threshold values between 0 and 1 
    thresholds = np.linspace(0, 1, thresholds_num)

    # Compute F1 score for each threshold by converting probabilities to binary predictions
    threshold_results = [
        f1_score(y_train, (y_pred >= t).astype(int)) for t in thresholds
    ]

    # Find the threshold with the highest F1 score
    best_threshold = thresholds[np.argmax(threshold_results)]

    return best_threshold


def get_scaler(args):
    if args.scale == 'minmax':
        return MinMaxScaler(random_state = RANDOM_STATE)
    else:
        return StandardScaler(random_state = RANDOM_STATE)
    
    
def get_resampler(strategy: str):
    """
    Returns a resampler or a pipeline of resamplers according to the strategy name.
    Does NOT apply fit_resample — avoids data leakage when used inside a CV pipeline.
    """
    from imblearn.over_sampling import SMOTE, KMeansSMOTE
    from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
    from imblearn.pipeline import Pipeline as imb_Pipeline
    
    if strategy == "none":
        return None

    elif strategy == "over-smote":
        return SMOTE(random_state=RANDOM_STATE)

    elif strategy == "over-kmeans":
        return KMeansSMOTE(
            cluster_balance_threshold=0.001,
            kmeans_estimator=20,
            n_jobs=-1
        )

    elif strategy == "under-random":
        return RandomUnderSampler(random_state=RANDOM_STATE)

    elif strategy == "under-kmeans":
        return ClusterCentroids(random_state=RANDOM_STATE)

    elif strategy == "both":
        # Placeholder target sizes for pipeline construction
        oversample = SMOTE(
            sampling_strategy='auto',
            k_neighbors=5,
            random_state=RANDOM_STATE
        )
        undersample = RandomUnderSampler(random_state=RANDOM_STATE)

        return imb_Pipeline([
            ('over', oversample),
            ('under', undersample)
        ])

    elif strategy == "both-kmeans":
        oversample = KMeansSMOTE(
            sampling_strategy='auto',
            random_state=RANDOM_STATE
        )
        undersample = ClusterCentroids(
            sampling_strategy='auto',
            random_state=RANDOM_STATE
        )
        return imb_Pipeline([
            ('over', oversample),
            ('under', undersample)
        ])

    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")
    

def get_model(args):
    if args.model == 'logistic':
        if False: # manually turn to true for cost sensitive model.
            logistic_params = {
                'random_state': RANDOM_STATE,
                'verbose': 1,
                'solver': 'lbfgs',
                'class_weight': {0: 25, 1: 75},  # Cost-sensitive weighting
                'C': 1.0,
                # 'max_iter': 1000,
            }

            return LogisticRegression(**logistic_params)

        logistic_params = {  # Modify manually
            'random_state': RANDOM_STATE,
            'verbose': 1,
            # 'C': 1.0,
            # 'solver': 'lbfgs',
            # 'max_iter': 1000,
        }
        return LogisticRegression(**logistic_params)

    elif args.model == 'rf':  # Random Forest
        rf_params = {  # Modify manually
            'random_state': RANDOM_STATE,
            'verbose': 1,
            # 'n_estimators': 100,
            # 'max_depth': None,
        }
        return RandomForestClassifier(**rf_params)

    elif args.model == 'knn':
        knn_params = {  # Modify manually
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'minkowski',
            # No random_state or verbose for KNN
        }
        return KNeighborsClassifier(**knn_params)

    else:  # MLP (Neural Network)
        mlp_params = {  # Modify manually
            'random_state': RANDOM_STATE,
            'verbose': True,
            # 'hidden_layer_sizes': (100,),
            # 'max_iter': 300,
        }
        return MLPClassifier(**mlp_params)
