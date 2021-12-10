# Term Deposit Subscription: Case Study Overview

- The objective of this project is to compare and fit various binary classifiers models in order to predict whether a client will subscribe to the term deposit based on the most relevant descriptive features.
- Goal is to identify the optimal model based on the best area under ROC curve metrics and allow the deployment of the most accurate model, allowing the bank to target the potential customer in their marketing campaign effectively.
- Cleaned and Preprocessed the Dataset.
- Performed Exploratory Data Analysis on the dataset to gain insights and trends in the data.
- Perform One-Hot Encoding for the categorical variables to be used for modeling.
- Optimized K Nearest Neighbors, Bayesian Naive Bayes, Decision Tree and Random Forest Classifier using GridSearchCV to reach the best model.
- A deep dive into data leakage to prevent target from leaking into models I chose for an accurate prediction.
- Performance measure used to evaluate the models is the Area Under the Receiver Operating Characteristic (RUC) Curve due to class imbalance in the dataset.


## References:
**Python Version:** 3.8<br/>
**Packages:** numpy, pandas, seaborn, matplotlib, Scikit-Learn<br/>
**Data Source:**  UCI Machine Learning Repository [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)<br/>
**Data Leakeage Article:** [Data Leakage in Machine Learning](https://towardsdatascience.com/data-leakage-in-machine-learning-6161c167e8ba)<br/>
**Standardize Age Group:** [Australian Bureau of Statistics](https://www.abs.gov.au/)<br/>

# Data Cleaning/Pre-processing

- 'Duration' column is removed as it highly affects the output target (e.g., if duration=0 then subscription='no')
- 'pdays' column is clamped into a binary feature (i.e., 0 = not previously contacted; 1=contacted)
- Descretizing the 'age' into standard Age Group (i.e., Children (0-14); Teen (15-24); Adult (25-64); Senior (65+))
- One-Hot Encoding performed on all the categorical features

# Exploratory Data Analysis

<p float="left">
    <img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/job_categories.png" width="35%" height="35%">
    <img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/relational-plot.png" width="40%" height="40%">
    <img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/correlation_matrix.png" width="30%" height="=30%">
</p>

# Model Building

The next steps is model building strategy and evaluate them. I started off by scaling the features using MaxMinScaler from the Scikit-learn module, then split the dataset into 70% train and 30% test set. 

I tried several different models and evaluated them using Area Under the Receiver Operating Characteristic (ROC) Curve. I chose the following metric because using the method as such will be robust to class imbalance.

Classifier I tried using Scikit-learn are:

- **K-Nearest Neighbors:** Lazy Learner as a baseline model

<img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/knn_performance.png" width="40%" height="40%">

- **Gaussian Naive Bayes:** Probability Based Learner as features will be independent to one another.

<img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/nb_performance.png" width="40%" height="40%">


- **Decision Tree:** Chosen with regards to the sparsity of the data and ideal for dealing with data containing multiple outliers.

    <img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/dt_performance_nottunned.png" width="40%" height="40%">

    - **Further Fine Tuning of Decision Tree Classifier**

    <img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/dt_performance_tunned.png" width="45%" height="45%">

- **Random Forest:** An emsemble that will be able to deal with the sparsity of data, I assume that it would be a good fit.

    <img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/rf_performance_nottunned.png" width="40%" height="40%">

    - **Further Fine Tuning of Random Forest Classifier**

    <img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/rf_performance_tunned.png" width="40%" height="40%">

# Model Performance

After performing hyperparameter tuning and further fine tuning on the models, The Random Forest Classifier far outperformed the other approaches on the test and validation sets.

For the **K-Nearest Neigbor (KNN)** classifier, we concluded that the best parameters are:
- n_neighbors = 7
- p = 1 (Manhattan)
- n_features = 50

The KNN Classifier resulted with a best score in terms of area under ROC curve of 0.634.

For the **Gaussian Naive Bayes (NB)** Classifier, we concluded that the best parameters are:
- n_features = 50
- var_smoothing = 0.9326

The Gaussian Naive Bayes (NB) Classifier resulted with a best score in terms of area under ROC curve of 0.710.


For the **Decision Tree (DT)** Classifier, we concluded that the best parameters are:
- max_depth = 10
- min_samples_split: 150
- n_features = 50

The Decision Tree (DT) Classifier resulted with a best score in terms of area under ROC curve of 0.654.

For the **Random Forest Tree (RFT)** Classifier, we concluded that the best parameters are:
- max_depth = 10
- n_estimators: 500
- n_features = 50

The Random Forest Tree (RFT) Classifier resulted with a best score in terms of area under ROC curve of 0.739.

The recall scores shows that the best model is Gaussian NB Classifier is the best model based on the Classification Report and the Confusion Matrix.

For our project, the model is best evaluate with area under ROC curve due to proportion imbalance. Hence, it is concluded Gaussian NB Classifier would be the best fitting model as it has the highest accuracy.

# Feature Importance

<img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/important_features.png" width="60%" height="60%">

# Critique and Limitations

The KNN algorithm used in this project is consider lazy learner. This algorithm does not derive any discriminative function from the training data and does not learn anything in the training period. KNN is relatively easy to implement as it only required two parameters, which is the value of k and the distance function. However, it does not work well with our data. Our dataset has high dimensions and it becomes difficult for the algorithm to calculate the distance in each dimension. Our dataset also has some noise and outliers which the KNN is sensitive to. Hence, in order to resolve this issue, a parameter search for k-values greater then 7 or higher need to be considered in future. 

The Naive Bayes Classifiers algorithm used in this project has a better performance if the assumption of independent predictors holds true. It is easy to implement and only requires a relatively small amount of training data to estimate. Naive Bayes implicitly assumes that all the attributes are mutually independent. However, full independence between events is quite rare. We have applied a blanket power transformation when hypertuning the Naive Bayes model, ignoring the irrelevant features within the dataset. This lead to the poor performance of the Naive Bayes model. In order to improve this performance, we should build a Gaussian Naive Bayes and Bernoulli Naive Bayes on the numerical and dummy descriptive features respectively.

Statistically, Decision Tree Classifier (DT) and the Random Forest Classifier (RFT) outperforms the other two classifier in terms of accuracy score. However, in order to improve the our results, other parameter such as `min_samples_leaf`, `max_features` can be expanded in future hyperparameter search. DT and RFT are simple and easy to understand. They are usually robust to outliers. DT and RFT generally lead to over-fitting of the data. Due to the over-fitting, it is highly likely of the high variance in the output leads to wrong predictions.

Our model split the sample into training and testing set with 70:30 ratio. The dataset has been cross-validated and repeated three times to improve performance. The performance metrics to score the model is based on area under the `ROC` curve. The reason is due proportion imbalance of the `target feature` in the data. A different metric can be applied as a performance indicator for our chosen models. This could potentially help improve the performance of the model as the overall accuracy score is looking at the fractions of correctly assigned positive and negative classes. 

# Conclusion

The descriptive features and target feature are split into a training set and a test set by a ratio of 70:30. The 70% of the data is used to build a classifier to evaluate the performance of the test set. The stratify option is set to the target feature in order to ensure to maintain a consistent ratio when splitting the data while the shuffle option is set to true in order to ensure randomness.

The split data is then fit to various classifier model, including K-Nearest Neighbor, Decision Tree, Random Forest and Gaussian Naive Bayes. Random Forest Importance (RFI) method is used with 100 estimators, 50 best descriptive features in each of the model are selected. Each of the algorithm has tuning with hyperparameters and further fine-tuning is done on both Decision Tree and Random Forest Classifier, then tested with best parameter. The performance of each of the classifier is evaluate using the area under ROC curve score method. Multiple runs is then performed in a cross-validation setting and then a paired t-test is conducted in order to determine whether the difference is statistically significant.


The Random Forest model with 50 of the best features selected by Random Forest Importance (RFI) produces the highest cross-validated AUC score on the training data. In addition, when evaluated on the test data (in a cross-validated fashion), the Random Forest and Decision Tree models outperforms both Naive Bayes and K-Nearest Neighbor models with respect to area under ROC curve.

However, the Naive Bayes model yields the highest recall score on the test data. We also observe that our models are fitted with the full features as selected by RFI when conditioned on the values of the hyperparameters in general. For this reason, it is deducted that 50 features are preferable, which potentially avoids overfitting and results in models that are easier to train and easier to understand.

In conclusion, Gaussian NB is identified as the optimal model based on the best area under ROC curve metrics and allow the deployment of the most accurate model, allowing the bank to target the potential customer in their marketing campaign effectively.


