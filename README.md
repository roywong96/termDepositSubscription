# Term Deposit Subscription: Case Study Overview

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
**Data Leakeage Article** [Data Leakage in Machine Learning](https://towardsdatascience.com/data-leakage-in-machine-learning-6161c167e8ba)<br/>

# Data Cleaning/Pre-processing

- Data is 

# Exploratory Data Analysis


# Model Building

The next steps is model building strategy. I started off by scaling the features using MaxMinScaler from the Scikit-learn module, then split the dataset into 70% train and 30% test set. 

I tried several different models and evaluated them using Area Under the Receiver Operating Characteristic (RUC) Curve. I chose the following metric because using the method as such will be robust to class imbalance.

Classifier I tried using Scikit-learn are:

- **K-Nearest Neighbors:** Lazy Learner as a baseline model

<img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/knn_performance.png" width="50" height="50">

- **Gaussian Naive Bayes:** Probability Based Learner as features will be independent to one another.


<img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/nb_performance.png" width="50" height="50">


- **Decision Tree:** Chosen with regards to the sparsity of the data and ideal for dealing with data containing multiple outliers.



- **Random Forest:** An emsemble that will be able to deal with the sparsity of data, I assume that it would be a good fit.


# Model Performance

After performing hyperparameter tuning and further fine tuning on the models, The Random Forest Classifier far outperformed the other approaches on the test and validation sets.

- **K-Nearest Neighbors:** ROC= 0.634
- **Gaussian Naive Bayes:** ROC= 0.710
- **Decision Tree:** ROC= 0.654
- **Random Forest:** ROC= 0.739




