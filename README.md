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
**Data Leakeage Article:** [Data Leakage in Machine Learning](https://towardsdatascience.com/data-leakage-in-machine-learning-6161c167e8ba)<br/>
**Standardize Age Group:** [Australian Bureau of Statistics](https://www.abs.gov.au/)<br/>

# Data Cleaning/Pre-processing

- Duration column is removed as it highly affects the output target (e.g., if duration=0 then subscription='no')
- pdays is clamped into a binary feature (i.e., 0 = not previously contacted; 1=contacted)
- Descretizing the age into standard Age Group (i.e., Children (0-14); Teen (15-24); Adult (25-64); Senior (65+))
- One-Hot Encoding performed on all the categorical features

# Exploratory Data Analysis

<img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/job_categories.png" width="30%" height="30%">
<img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/relational-plot.png" width="50%" height="50%">
<img src="https://github.com/roywong96/termDepositSubscription/blob/master/images/correlation_matrix.png" width="30%" height="=30%">

# Model Building

The next steps is model building strategy. I started off by scaling the features using MaxMinScaler from the Scikit-learn module, then split the dataset into 70% train and 30% test set. 

I tried several different models and evaluated them using Area Under the Receiver Operating Characteristic (RUC) Curve. I chose the following metric because using the method as such will be robust to class imbalance.

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

- **K-Nearest Neighbors:** ROC= 0.634
- **Gaussian Naive Bayes:** ROC= 0.710
- **Decision Tree:** ROC= 0.654
- **Random Forest:** ROC= 0.739

# Feature Importance



# Conclusion




