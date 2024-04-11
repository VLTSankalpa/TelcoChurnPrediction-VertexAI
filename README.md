# Telco Customer Churn Prediction
The goal of this project is to predict customer churn for a telecommunications company. The dataset contains information about customers, including their demographics, services they subscribe to, account information, and whether they churned or not. The project involves exploratory data analysis (EDA), data visualization, feature engineering, and data preprocessing to prepare the data for modeling.

Expected outcome of these process is to have a clean, well-understood dataset ready for feature engineering and model development. All the steps will be documented and explained in a Jupyter notebook. The project will involve identifying and handling missing values, encoding categorical variables, scaling numerical features, and splitting the data into training, validation, and test sets. The data will be saved to an `.npz` file, which can then be loaded for training a machine learning model.

After the data is cleaned and prepared, the project will involve training, tuning and deploying a machine learning model to predict customer churn using Google Cloud Vertex AI. The model will be evaluated using metrics such as accuracy, precision, recall, and F1 score. The project will also involve identifying important features that contribute to customer churn and providing recommendations to reduce churn rate.

# **Exploratory Data Analysis (EDA)**
- [x] List of Columns
- [x] Dataset Shape
- [x] Data Types
- [x] List all unique values in each column
- [x] Convert data types of columns
- [x] Handling missing values
- [x] Summary Statistics of numeric columns
# **Data Visualization**
- [x] Kernel Density Estimate (KDE) plots
- [x] Q-Q plots
- [x] Histograms
- [x] Boxplots
- [x] Scatter plots
- [x] Heatmaps
- [x] Count plots
# **Feature Engineering**
-  [x] **`AverageMonthlyCharges`**: It's common for customers to have variations in their charges throughout their tenure. This feature represents the average spend per month.
- [x] **`TenureGroups`**: Grouping tenure into categorical bins could reveal patterns related to customer loyalty and churn rate.
# Identify outliers
- [x] IQR Method
- [x] Z-score Method
# **Encode Categorical Variables**
- [x] Encode binary variables (**`gender`**, **`Partner`**, **`Dependents`**, **`PhoneService`**, **`PaperlessBilling`**, **`Churn`**) with 0 and 1.
- [x] Use one-hot encoding for nominal variables with more than two categories (**`MultipleLines`**, **`InternetService`**, **`OnlineSecurity`**, **`OnlineBackup`**, **`DeviceProtection`**, **`TechSupport`**, **`StreamingTV`**, **`StreamingMovies`**, **`Contract`**, **`PaymentMethod`**, **`TenureGroups`**) to prepare them for modeling.
- [x] Scale Numerical Features: Standardize or normalize **`AverageMonthlyCharges`**,**`tenure`**, **`MonthlyCharges`**, and **`TotalCharges`**.
# **Training Data Preparing**
- [x] Splits the data into feature (**`X`**) and label (**`y`**) arrays.
- [x] Uses **`train_test_split`** twice to create a train set (60% of the data), a validation set (20%), and a test set (20%).
- [x] Saves the training, validation, and test sets to an **`.npz`** file, which can then be loaded for training.
# Machine Learning Model Development

The goal is to select and prototype suitable machine learning algorithms for predicting customer churn for a subscription-based telco service. This involves evaluating various models to identify the most effective approach for this specific churn prediction task.

## Initial Model Prototyping

Several models were prototyped to assess their suitability and performance for the churn prediction task. These models can be built using standard libraries with minimal effort. If the dataset and preprocessing required vary significantly from one model to another, resulting in considerable training effort, we must stick to theoretical concepts. This approach involves selecting a few ML algorithms well-suited for the task and limiting the number of models tried. But in this case following models were prototyped:


- [x] **Logistic Regression Model Prototyping**
- [x] **Random Forest Model Prototyping**
- [x] **XGBoost Model Prototyping**
- [x] **DNN Model Prototyping**
- [x] **CNN for Tabular Data Prototyping**

## Evaluation Metrics

For each prototyped model, several key metrics were considered to evaluate performance, including accuracy, precision, recall, and the confusion matrix. These metrics provide a comprehensive view of each model's strengths and weaknesses in predicting customer churn. Based on those metrics, the best models for Vertex AI Vizier hyperparameter tuning will be selected.

# **Vertex AI Training, Tuning, and Deployment**
- [ ]  **Training XGBoost Model on Vertex AI**: Train the XGBoost Model on Vertex AI as a custom training job.
- [ ]  **Training DNN Model on Vertex AI**: Train the DNN Model on Vertex AI as a custom training job.
- [ ]  **Tuning XGBoost Model on Vertex AI**: Tune the XGBoost Model on Vertex AI using Vizier hyperparameter tuning.
- [ ]  **Tuning DNN Model on Vertex AI**: Tune the DNN Model on Vertex AI as part of a custom training job.
- [ ]  **Deployment**: Deploy the model as Vertex AI model endpoints for predictions.
