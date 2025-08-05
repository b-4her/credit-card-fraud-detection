## Model Results

| Model                          | Dataset  |   Train F1 |   Validation F1 |
|:------------------------------|:---------|-----------:|----------------:|
| Logistic (base)               | Sample   |     0.7143 |          0.7083 |
| Logistic (base)               | Original |     0.7169 |          0.7251 |
| Logistic (best threshold)     | Original |            |          0.7179 |
| Logistic (expansion=1)        | Sample   |     0.7714 |          0.8333 |
| Logistic (expansion=2)        | Sample   |     0.7097 |          0.5882 |
| Logistic (expansion=3)        | Sample   |     0.6552 |          0.5    |
| Logistic (cost-sensitive)     | Original |     0.7838 |          0.8261 |
| Logistic (SMOTE)              | Original |     0.7401 |          0.7487 |
| Logistic (KMeans-SMOTE)       | Original |     0.6545 |          0.7027 |
| Logistic (Random Under)       | Original |     0.5473 |          0.5414 |
| Logistic (KMeans Under)       | Original |     0.7524 |          0.7805 |
| Logistic (Random + SMOTE)     | Original |     0.7717 |          0.8111 |
| Logistic (PCA)                | Original |     0.8034 |          0.8070 |
| Logistic (best overall manual)| Original |     0.8065 |          0.8118 |
| Neural Network (manual)       | Original |     0.9693 |          0.8606 |
| Random Forest (manual)        | Original |     1      |          0.881  |
| KNN (PCA)                     | Original |     0.8689 |          0.8639 |
| KNN (no PCA)                  | Original |     0.8638 |          0.8415 |
| Logistic (random search)      | Original |     0.8212 |          0.795  |
| NN (random search)            | Original |     0.9322 |          0.8383 |
| RF (random search)            | Original |     0.9562 |          0.8475 |
| KNN (PCA, random search)      | Original |     1      |          0.7979 |
| KNN (no PCA, random search)   | Original |     1      |          0.7629 |
| Voting Classifier             | Original |     0.9983 |          0.8824 |
| KNN (PCA)                     | Test (final) |     - |          0.8691 |
| Voting Classifier             | Test (final) |     - |          0.8557 |


> **Note:**  
> The final models from each phase are saved in the `models/` directory and can be evaluated using the [`saved_models_evaluator.py`](../scripts/saved_models_evaluator.py) script located in the `scripts/` folder.  
>
> For detailed insights into model selection, feature engineering decisions, and performance analysis, refer to the 📄 [`final report`](summary/fraud-detection-summary-report.pdf) in the `summary/` directory.