# Credit Cards Customers Project

## Project Description
Customer Churn (customer attrition) is the most challenge problem for business such as credit cards or telecommunication companies etc. Building models to predict who is going to churn would help improve business, and companies can prevent from losing their customers. 

In this project, I analyzed [credit card customers' dataset](https://www.kaggle.com/sakshigoyal7/credit-card-customers) and built machine learning (ML) models to predict who churn the service. More details for the dataset is [here](https://leaps.analyttica.com/sample_cases/11). I also wrote a [blog](https://medium.com/@yejizoeseoung/credit-card-customers-analysis-6a193f00c044) for this project.


## Machine Learning Models
We compared 6 models (Logistic Regression, Support Vector Machine, KNeighbor, Random Forest (RF), AdaBoost (ADA), and XGBoost (XGB)) by using results' matrics (Accuracy, Recall, Precision, and ROC AUC score). Given the results, we chose three models (RF, ADA, XGB) and tuned hyperparameters to find best parameters. 

XGBoost model showed best results compared to other models. 
![ML1](/images/MLmodels.png)



## Visualization
Creating graphs help solve the questions below:

1. How many people decided to quit the credit card service?
The dataset has unbalanced data whether customers churn or retain. The majority of customers retain the service. 
![graph1](/images/churn.png)



2. Any differences between Churn and Exist groups depending on each feature?
Some features show differences between Churn and Exist groups.


![graph2](/images/re_bal.png)
![graph3](/images/trans_ct.png)
![graph4](/images/cat_graphs.png)


3. Are there correltions between numerical variables?
![graph5](/images/corr.png)


## Dependencies
- Python 3.5+
- Machine Learning Libraries: Numpy, Pandas, Sciki-Learn
- Visualization: Matplotlib


## File Description
The files structure is arranged as below:

    - Data
        - BankChurners.csv: raw data 
        - df_clean.csv: cleaned dataset to analyze and build ML models

    - Credit_Card_Customers.ipynb shows the entire workflow including cleaning data, visualizing data, building the models, and evaludating models.
    - readme.md