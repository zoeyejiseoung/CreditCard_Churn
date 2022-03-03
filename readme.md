# Credit Cards Customers Project

## Project Description
Customer Churn (customer attrition) is the most challenge problem for business such as credit cards or telecommunication companies etc. Building models to predict who is going to churn would help improve business, and companies can prevent from losing their customers. 

In this project, I analyzed [credit card customers' dataset](https://www.kaggle.com/sakshigoyal7/credit-card-customers) and built machine learning (ML) models to predict who churn the service. More details for the dataset is [here](https://leaps.analyttica.com/sample_cases/11). I also wrote a [blog](https://medium.com/@yejizoeseoung/credit-card-customers-analysis-6a193f00c044) for this project.


## Machine Learning Model Pipeline
We compared 6 models (Logistic Regression, Support Vector Machine, KNeighbor, Random Forest (RF), AdaBoost (ADA), and GradientBoostingModel (GBM)) by using results' matrics (Accuracy, Recall, Precision, and ROC AUC score). Given the results, we built GBM model pipeline by tuning hyperparameters. 


GBM model showed best results compared to other models. 
![ML1](/images/MLmodels.png)

**By using the best hyperparamters, we got an improved GBM model showing 7% increased performance (0.993 of ROC-AUC score) compared to the baseline (0.916 of ROC-AUC).**

Tuned GBM model shows better performance compared to the baseline GBM model. 
![ML2](/images/GBMCompare.png)


**NOTE**: Please see [Building_Model_Pipeline.ipynb](https://github.com/yejiseoung/CreditCard_Churn/blob/main/Building_Model_Pipeline.ipynb), if you want to see more detail.


## Dealing with Imbalanced dataset
![graph1](/images/churn.png)
We have a classification problem. We have 83% of Existing customer data and 16% of Attrited customer data. Based on the percentage of the distribution of target, we can say that this dataset is imbalanced. So, we needed to deal with this very carefully.

I compared 5 different over-sampling methods: RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, and SVMSMOTE.
As a result, ADASYN and BorderlineSMOTE showed the best performances so that I chose ADASYN to balance the dataset in the final pipeline. 
![graph2](/images/oversampling.png)

**NOTE**: Please see [Oversampling.ipynb](https://github.com/yejiseoung/CreditCard_Churn/blob/main/Oversampling.ipynb), if you want to see more detail.


## Visualization_EDA 
Creating graphs help solve the questions below:


1. Any differences between Churn and Exist groups depending on each feature?
Some features show differences between Churn and Exist groups.


![graph2](/images/re_bal.png)
![graph3](/images/trans_ct.png)
![graph4](/images/cat_graphs.png)


2. Are there correlations between numerical variables?
![graph5](/images/corr.png)


**NOTE**: Please see [Cleaning_EDA_and_Visualization.ipynb](https://github.com/yejiseoung/CreditCard_Churn/blob/main/Cleaning_EDA_and_Visualization.ipynb), if you want to see more detail.




## File Description
The files structure is arranged as below:

    - Data
        - BankChurners.csv: raw data 
        
    - Building_Model_Pipeline.ipynb shows the workflow regarding building Gradient Boosting Model.
    - Cleaning_EDA_and_Visalization.ipynb shows data cleaning, exploratory data analysis, and visualizing data.
    - FeatureEngineeringScaling.ipynb shows investigations regarding feature engineering, feature scaling, and feature importance. 
    - HyperparameterTuningForGBM.ipynb shows hyperparameter tuning by using Optuna to find the best parameters for GBM model. 
    - Oversampling.ipynb shows finding the best over-sampling methods for this project to deal with imbalanced dataset.
    - readme.md


## Dependencies
- Python 3.5+
- Machine Learning Libraries: Numpy, Pandas, Sciki-Learn
- Visualization: Matplotlib


## Acknowledgements
Data was provided by Kaggle