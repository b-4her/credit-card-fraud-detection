import pandas as pd
import time

from sklearn.pipeline import Pipeline as normal_Pipeline
from imblearn.pipeline import Pipeline as imb_Pipeline

from model_utils_data import preprocessing_pipeline, get_args, make_config, save_model_and_config

from model_utils_eval import get_scaler, get_model, evaluate_model, find_best_threshold, search_and_train, get_resampler

# Should be modidified manually before saving a model
MODEL_FILE = 'best_vc_rndm_search'
CONFIG_FILE = 'best_vc_rndm_search'

DEFAULT_SAMPLE_PATH = '../data/sample/'
DEFAULT_PATH = '../data/split/'

def main():
    args = get_args()

    path = DEFAULT_PATH
    if args.use_sample:
        path = DEFAULT_SAMPLE_PATH

    train_path = path + "train.csv"
    val_path = path + "val.csv"

    if args.test:  
        # train on both train and val 
        train_path = path + "trainval.csv"
        val_path = path + "test.csv"   

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    if args.preprocess:
        train_df = preprocessing_pipeline(train_df)
        val_df = preprocessing_pipeline(val_df)

    X_train = train_df.drop(columns='Class')
    y_train = train_df['Class']

    X_val = val_df.drop(columns='Class')
    y_val = val_df['Class']

    # Build steps (for the pipeline)
    steps = []

    if args.resampler != "none":
        print("Adding resampler to the pipeline...")
        resampler = get_resampler(args.resampler)
        if args.resampler == 'both' or args.resampler == 'both-kmeans':
            steps.append(resampler[0])
            steps.append(resampler[1])
        else:
            steps.append(('resampler', resampler))

    if args.expansion > 0:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.compose import ColumnTransformer

        print("Adding feature expansion to the pipeline...")

        # Selecting cols based on importance in the RF tree during the EDA (Top 6)
        expansion_columns = ['V17', 'V14', 'V12', 'V6', 'V10', 'V21'] 

        # Create the ColumnTransformer
        poly_transformer = ColumnTransformer(
            transformers=[
                ('poly', PolynomialFeatures(degree=args.expansion, include_bias=False), expansion_columns)
            ],
            remainder='passthrough'
        )

        steps.append(('poly_expansion', poly_transformer))

    # Add PCA if needed (After resampling & expansion)
    if args.pca:
        from sklearn.decomposition import PCA
        print("Adding PCA to the pipeline...")
        pca = PCA(n_components=20)
        steps.append(('pca', pca))

    # Add scaler (After PCA & Expansion - to ensure featuers are fully scaled)
    if args.scale != 'none':
        scaler = get_scaler(args)
        print("Adding scaler to the pipeline...")
        steps.append(('scaler', scaler))

    # Track training time
    start_time = time.time()

    if args.random_search or args.model=='vc':
        model_pipeline = search_and_train(X_train, y_train, steps, args)

    else:
        # Add model
        model = get_model(args)
        steps.append(('model', model))

        # Build pipeline
        print("\nFitting pipeline...")
        if args.resampler == 'none':
            model_pipeline = normal_Pipeline(steps=steps, verbose=True)
        else:
            model_pipeline = imb_Pipeline(steps=steps, verbose=True)
        
        model_pipeline.fit(X_train, y_train)

    end_time = time.time()

    # Report training duration
    training_time = end_time - start_time
    print(f"\nModel training completed in {training_time:.2f} seconds.\n")


    if args.thresholds_num > 0:
        threshold = find_best_threshold(model_pipeline, X_train, y_train, args.thresholds_num)
        print(f"Best threshold is {threshold:.4f}")
    else:
        threshold = 0.5
        print("Skipping threshold search, using default threshold 0.5")

    model_results = evaluate_model(model_pipeline, threshold, X_train, y_train, X_val, y_val)


    if args.save_model:
        model_info = {
                    "model_pipeline": model_pipeline,
                    "threshold": threshold,
                 }

        config_file = make_config(args, model_results, threshold, training_time, MODEL_FILE)
        save_model_and_config(model_info, config_file, MODEL_FILE, CONFIG_FILE)


if __name__ == "__main__":
    main()