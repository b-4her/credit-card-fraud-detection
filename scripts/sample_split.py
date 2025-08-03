import pandas as pd
from sklearn.model_selection import train_test_split


def take_sample(df, is_train=True, sample_size=20000, random_state=42):
    """
    Sample from the same distribution.

    - If is_train=True: returns a stratified sample of size `sample_size`.
    - If is_train=False: splits a stratified sample into validation and test sets.
    """
    if is_train:
        sample, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df['Class'],
            random_state=random_state
        )
        return sample
    else:
        # Take a stratified sample first
        sample, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df['Class'],
            random_state=random_state
        )
        # Split that sample into validation and test (70/30)
        val, test = train_test_split(
            sample,
            test_size=0.3,
            stratify=sample['Class'],
            random_state=random_state
        )
        return val, test


if __name__ == "__main__":
    DIRECTORY_PATH = "../data/split/"
    SAMPLE_PATH = "../data/sample/"

    df_train = pd.read_csv(DIRECTORY_PATH + "train.csv")
    df_val = pd.read_csv(DIRECTORY_PATH + "val.csv")

    train_sample = take_sample(df_train)
    val_sample, test_sample = take_sample(df_val, is_train=False)

    # combine train & val in one w
    train_val_sample = pd.concat([train_sample, val_sample], axis=0)

    # Save samples into csv
    train_sample.to_csv("../data/sample/train.csv", index=False)
    val_sample.to_csv("../data/sample/val.csv", index=False)
    train_val_sample.to_csv("../data/sample/trainval.csv", index=False)
    test_sample.to_csv("../data/sample/test.csv", index=False)
