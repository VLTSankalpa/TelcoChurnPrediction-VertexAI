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

