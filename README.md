# credit-card-fraud-detection
TODO

add toggle list later on secitons you want to include like report and CLI docs
For this project no need for api, (project 1 focuses on EDA & API e2e idea)
this project should just focus on modeling and creating descaent cli for that, explore as many techniques and analyze them.

proj overview: (add results of baseline and final model later for nice comparison) update it later and add/remove stuff 
(maker shorter version for the about section)

The primary goal of this project is to build an effective fraud detection model that optimizes the **F1-Score**, which balances both precision and recall. This is especially important in the context of credit card fraud detection, where the cost of missing fraudulent transactions or falsely flagging legitimate ones can be high.

The dataset is highly imbalanced, with legitimate transactions greatly outnumbering fraudulent ones—a common trait in real-world financial data. Additionally, most of the features have been transformed using PCA to protect user privacy, resulting in anonymized and less interpretable variables. Because of this, the exploratory data analysis (EDA) will be brief and focused mainly on understanding the basic structure of the data and class distribution.

Most of the effort in this project will be focused on the modeling phase. To address the imbalance and improve model performance, a combination of supervised and unsupervised learning techniques will be used. These include traditional classifiers like logistic regression and random forests, along with anomaly detection methods such as isolation forest and one-class SVM. Techniques like oversampling, undersampling, and class weighting will also be explored to mitigate the class imbalance.

The overall objective is to develop a reliable and generalizable model that can effectively detect fraudulent transactions under realistic constraints while maintaining a strong F1-Score.


Explain:
project is two sections
1. supervised techniuqes (part 1 of the project)
2. unsupervised techniques (part 2 of the project)
3. summary and comparison between both sections