<a id="readme-top"></a>

<!-- PROJECT TITLE -->
<br />
<div align="center">
  <h1 align="center"><b>Credit Card Transaction Fraud Detection</b></h1>

  <p align="center">
    <i>This project tackles credit card fraud detection using machine learning, focusing on the challenge of extreme class imbalance. The main goal is to model a real-world problem and learn how to navigate different development phases—from preprocessing and EDA to experimentation, tuning, and evaluation—until reaching the best possible model.</i>
    <br />
    <a href="https://youtu.be/your-demo-link"><strong>Quick Demo</strong></a>
  </p>
</div>

---
<!-- TABLE OF CONTENTS -->
<details>
<summary><strong>Table of Contents</strong></summary>

- [Overview](#overview)
- [Repo Structure and File Descriptions](#repo-structure-and-file-descriptions)
- [Modeling Workflow](#modeling-workflow)
  - [EDA](#eda)
  - [Modeling and Results](#modeling-and-results)
  - [Final Model Performance](#final-model-performance)
- [Key Findings](#key-findings)
- [Project Report](#project-report)
- [Training and Evaluating Models](#training-and-evaluating-models)
  - [Setup and Installation](#setup-and-installation)
  - [Running the CLI](#running-the-cli)
  - [Evaluating Saved Models](#evaluating-saved-models)
- [Contact Information](#contact-information)
- [Resources and Credits](#resources-and-credits)

</details>

---

## Overview

The main objective of this project is to build a robust end-to-end machine learning pipeline for fraud detection, with a focus on handling imbalanced data and evaluating different modeling strategies. The process begins with exploratory data analysis (EDA) to understand feature importance and distribution patterns. Emphasis is placed on feature preprocessing, expansion, and transformation to improve model performance.

A variety of models were trained and compared using metrics suited for imbalanced classification, with F1-score chosen as the primary evaluation metric due to its balance between precision and recall. Techniques like resampling (SMOTE, undersampling), cost-sensitive learning, and ensemble methods were applied to better detect rare fraudulent cases. The workflow is modular and configurable via command-line arguments, supporting experimentation with different models, thresholds, and pipeline settings. This structure enables consistent testing, reproducibility, and extensibility for further development.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Repo Structure and File Descriptions

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── config/                          # Hyperparameter configurations
│   ├── best_knn_no_search.json
│   └── best_rf_rndm_search.json    # 
├── models/                          # Trained models 
│   ├── best_knn_no_search.pkl
│   └── best_rf_rndm_search.pkl
├── notebooks/
│   └── EDA.ipynb                    # Exploratory Data Analysis notebook
├── scripts/                         # Training and evaluation scripts
│   ├── model_train.py
│   ├── model_utils_data.py
│   ├── model_utils_eval.py
│   ├── sample_split.py
│   └── saved_models_evaluator.py
├── summary/                         # Final project outputs and results
│   ├── fraud-detection-project-report.pdf
│   └── model_results.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Modeling Workflow

### EDA

- Explored feature distributions and class imbalance.
- Identified key features related to fraud using correlation and importance analysis.
- Based on EDA insights, I opted to emphasize model-selection and stratified sampling instead of detailed feature engineering.

See the full analysis in the 📄 [`EDA Notebook`](notebooks/eda.ipynb).

### Modeling and Results

- Conducted multiple modeling phases:
  - Manual tuning
  - Random search
  - Voting classifier
- Applied techniques like resampling, feature expansion, and scaling.
- Special attention was paid to threshold tuning and evaluation using F1 and average precision scores.

### Final Model Performance

| Model              | Test F1  | Avg Precision |
|-------------------|----------|----------------|
| KNN (Manual Tuning)      | 0.8691   | 0.8293         |
| Voting Classifier (Random Search) | 0.8557   | 0.8694         |

> Full results for all tested models can be found in 📄 [`model_results.md`](summary/model_results.md).

Both KNN and Voting Classifier were selected as the final models after extensive evaluation. KNN achieved the highest F1 score on the test set, making it more suitable for maximizing correct fraud detection. On the other hand, the Voting Classifier yielded a higher average precision, indicating better ranking performance and confidence in its fraud predictions. Depending on the specific deployment priorities either model can be considered for production.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Key Findings

- Understanding model mechanics (e.g., why scaling helps NN) is essential.
- Manual tuning can outperform automated search in complex or time-limited scenarios.
- Testing multiple modeling techniques is key to discovering top-performing models.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Project Report


A detailed summary of the project is available in the `summary/` folder or 📄 [`final-report`](summary/fraud-detection-project-report.pdf).  
It includes the full analysis, methodology, results, and additional insights beyond what's covered in this README.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Training and Evaluating Models

### Setup and Installation

1. **Clone the repository**

  ```bash
  git clone https://github.com/b-4her/credit-card-fraud-detection.git
  ```
2. **Navigate to the project directory**

  ```bash
  cd credit-card-fraud-detection
  ```

3.	**Install dependencies**

   ```bash
   pip install -r requirements.txt
  ```

## Running the CLI

The CLI provides flexible options to configure the training pipeline, including model selection, preprocessing, resampling strategies for imbalanced data, feature expansion, PCA, and evaluation settings.

### Important

Before running the script, make sure to **manually update** the following variables in the script if needed:

- `MODEL_FILE = 'best_vc_rndm_search'`  
- `CONFIG_FILE = 'best_vc_rndm_search'`  
- `DEFAULT_SAMPLE_PATH = '../data/sample/'`  
- `DEFAULT_PATH = '../data/split/'`

These control the **model file name**, **config file name**, and **data source paths**. You should change them manually **before saving a model or training with different data or configurations** to avoid overwriting files or using incorrect data.

### Key Arguments

- `--model`: Choose the model to train (`logistic`, `nn`, `rf`, `knn`, or `vc`).
- `--scale`: Data scaling method (`minmax`, `standardization`, or `none`).
- `--resampler`: Resampling approach to handle class imbalance (e.g., `none`, `over-smote`, `under-random`, `both`, etc.).
- `--expansion`: Degree of polynomial feature expansion (0 to disable).
- `--pca`: Apply PCA for dimensionality reduction (18 components by default).
- `--thresholds-num`: Number of thresholds to evaluate for best classification threshold.
- `--test`: Train on train+val and evaluate on the test set.
- `--preprocess`: Apply preprocessing to data.
- `--save-model`: Save the trained model to disk.
- `--use-sample`: Use a smaller sample dataset.
- `--random-search`: Enable random search hyperparameter tuning.

### Example usage

First, navigate to the `scripts` directory where the CLI script is located:

```bash
cd scripts
```

```bash
python3 model_train.py --model=rf --scale=standardization --resampler=over-smote \
--expansion=2 --pca --thresholds-num=100 --preprocess --save-model
```

This runs a Random Forest with standard scaling, SMOTE oversampling, polynomial features of degree 2, PCA 18 components, threshold search with 100 candidates, preprocessing enabled, and saves the model.

### Evaluating Saved Models

This script is designed to **load a previously saved model** and evaluate its performance on the test dataset. You can control which model to load in **two ways**:

1. **Modify the constant variables in the script**:
   - `DEFAULT_MODEL_NAME`: Name of the model file (e.g., `'best_nn_no_search.pkl'`)
   - `DEFAULT_MODEL_DIR`: Directory where models are stored
   - `DEFAULT_TEST_FILE`: Path to the test CSV file

2. **Use the command line to pass the model name**:
   - This overrides the default model name at runtime.
   - Example:
     ```bash
     python evaluate_model.py --model my_trained_model.pkl
     ```

Once the model is loaded, it uses the pipeline and threshold saved in the pickle file to:

- Preprocess the test data
- Predict probabilities and labels
- Print metrics: **F1 Score**, **Precision**, **Recall**, and **Average Precision**

This provides a quick way to compare different saved models on the same test set.

> **Note:** The final model (`Voting Classifier`), saved as `best_vc_rndm_search.pkl`, is provided as a **zipped file** due to its large size. **Unzip it before use.**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

### Contact Information
For any questions or feedback, reach out via:
- LinkedIn: [b-4her](https://www.linkedin.com/in/b-4her)
- GitHub: [b-4her](https://github.com/b-4her)
- YouTube: [b-4her](https://www.youtube.com/@b-4her)
- Email: baher.alabbar@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Resources and Credits

* The dataset used in this project is private and cannot be publicly shared.
* Some code snippets and implementation ideas were refined with the help of ChatGPT. Feedback is welcome on any part of the codebase.
* All analysis and insights presented in this project are entirely my own.

### Libraries and Frameworks:

* **scikit-learn** – Machine learning models and preprocessing.
* **pandas** – Data manipulation and cleaning.
* **numpy** – Numerical computations.
* **seaborn**, **matplotlib** – Visualization.
* **argparse** – CLI interface.

### Development Tools:

* **Anaconda** – Environment and package management.
* **Jupyter Notebooks** – Interactive development and testing.
* **VS Code** – Code editing and debugging.
* **GitHub** – Version control and collaboration.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---


