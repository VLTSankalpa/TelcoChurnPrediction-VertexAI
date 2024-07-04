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

# Cloud-Native API Development for Customer Churn Prediction Model

This phase encompasses the development and deployment of a customer churn prediction model as a cloud-native API. The model, trained using XGBoost and saved as a JSON file, is integrated into a microservices architecture. The model is a binary classifier that predicts whether a customer will churn (leave the service) or not. Each microservice—pre-processing, inference, and post-processing—handles specific tasks, ensuring modularity and efficiency. The deployment utilizes Kubernetes for orchestration, facilitating scalability, load balancing, and high availability. The system follows best practices for asynchronous request handling, security, and monitoring, resulting in a robust and efficient API for real-time customer churn predictions.

## Technical Requirements

1. **Model and Artifacts**:
    - **XGBoost Model**: A binary classifier saved as a JSON file, trained to predict customer churn.
    - **Label Encoder**: A tool used to encode categorical features into numerical values, saved as a pickle file.
    - **MinMax Scaler**: A scaler used to normalize features to a range, saved as a pickle file.
2. **Microservices**:
    - **Pre-processing Service**:
        - **Function**: Prepares the input data for the model by loading and applying the label encoder and MinMax scaler.
        - **Details**: Loads the label encoder and MinMax scaler artifacts from pickle files. Transforms the incoming data accordingly.
    - **Inference Service**:
        - **Function**: Performs model inference using the pre-trained XGBoost model.
        - **Details**: Loads the XGBoost model from a JSON file. Receives pre-processed data and outputs predictions as binary values (1s and 0s).
    - **Post-processing Service**:
        - **Function**: Converts model predictions into human-readable format.
        - **Details**: Translates binary predictions (1s and 0s) into "churn" or "non-churn". Provides the final output to the client.

## Functional Requirements

1. **Asynchronous Request Handling**:
    - **Requirement**: The system must handle multiple requests concurrently to ensure efficient processing and responsiveness.
    - **Implementation Details**:
        - Each microservice should be designed to handle I/O operations in a non-blocking fashion. This involves using asynchronous web frameworks (like FastAPI) that support asynchronous endpoints.
        - These endpoints use the async/await pattern to perform non-blocking I/O operations, allowing other tasks to be processed simultaneously.
        - Example: When a request involves reading data from a disk or making a network call, the service does not wait for the operation to complete. Instead, it moves on to handle other requests, returning to the original task when the I/O operation completes.
    - **Why**: Asynchronous handling maximizes resource utilization and improves the system’s ability to process multiple requests concurrently, thereby reducing latency and increasing throughput.
2. **Microservices Architecture**:
    - **Requirement**: The system must be divided into three main microservices: pre-processing, inference, and post-processing.
    - **Implementation Details**:
        - **Pre-processing Service**: Handles tasks like data scaling, encoding, and initial transformations. It loads necessary artifacts (like scalers and encoders) at startup and applies these transformations to incoming data.
        - **Inference Service**: Loads the trained model at startup and performs predictions on the pre-processed data. It receives transformed data from the pre-processing service and outputs the prediction results.
        - **Post-processing Service**: Handles tasks like decoding the model output and formatting the results for the client. It loads necessary artifacts (like label encoders) at startup and applies these to the model predictions.
    - **Why**: Microservices architecture promotes separation of concerns, modularity, and reusability. It allows each service to be developed, tested, and scaled independently, making the system more manageable and flexible.
3. **Scalability**:
    - **Requirement**: The system must scale horizontally to handle increased loads.
    - **Implementation Details**:
        - Use Kubernetes to orchestrate and manage the deployment of microservices. Kubernetes monitors CPU and memory usage of each instance and automatically scales the number of instances up or down based on predefined thresholds.
        - Configure the Horizontal Pod Autoscaler (HPA) to adjust the number of replicas of each microservice based on metrics like CPU utilization, request count, and request latency.
        - Multiple instances of each microservice ensure that load is evenly distributed and no single instance becomes a bottleneck.
    - **Why**: Horizontal scaling ensures that the system can handle high traffic volumes by adding more instances of the microservices, distributing the load, and maintaining performance.
4. **Load Balancing**:
    - **Requirement**: Efficiently distribute incoming requests across multiple instances of each microservice.
    - **Implementation Details**:
        - Use NGINX as the external load balancer to distribute incoming requests to the API Gateway.
        - The API Gateway (e.g., Kong or Istio) routes requests to the appropriate microservices and provides internal load balancing and service discovery.
        - Within each instance, multiple worker processes (e.g., managed by Uvicorn or Gunicorn) handle requests concurrently.
    - **Why**: Load balancing ensures that no single instance or worker is overwhelmed with requests, distributing the load evenly to maintain system stability and performance.
5. **Performance**:
    - **Requirement**: Minimize latency and ensure requests are processed within acceptable time limits.
    - **Implementation Details**:
        - Optimize asynchronous operations to ensure all I/O-bound tasks are non-blocking.
        - Configure proactive scaling policies to add instances before CPU utilization becomes critical, using custom metrics like request latency and queue length to trigger scaling.
    - **Why**: Minimizing latency is crucial for maintaining a responsive system, especially under high load conditions. Proactive scaling prevents bottlenecks and ensures timely processing of requests.
6. **Security**:
    - **Requirement**: Ensure that all API endpoints are secure and authenticated.
    - **Implementation Details**:
        - Implement authentication mechanisms such as OAuth2 or OpenID Connect to secure API endpoints. Validate access tokens before processing requests.
        - Use API Gateway features to enforce security policies and manage access control.
    - **Why**: Security is essential to protect sensitive data and prevent unauthorized access to the system.
7. **Multiple Workers**:
    - **Requirement**: Each microservice instance must handle concurrent requests using multiple worker processes.
    - **Implementation Details**:
        - Each instance of a microservice runs multiple worker processes (e.g., Uvicorn or Gunicorn workers). This allows the instance to handle multiple requests in parallel.
        - Each worker process uses asynchronous programming to handle I/O-bound tasks concurrently, enabling efficient utilization of resources.
        - Example: If a pre-processing service instance has 4 workers and receives 33 requests, each worker handles approximately 8-9 requests asynchronously.
    - **Why**: Using multiple workers within each instance enhances the ability to handle a high volume of concurrent requests, improving throughput and reducing response times.

## Non-Functional Requirements

1. **Reliability**:
    - **Requirement**: Ensure high availability and fault tolerance.
    - **Implementation Details**:
        - Use Kubernetes to manage service instances and perform health checks. Kubernetes can automatically restart failed instances and distribute traffic to healthy instances.
        - Implement redundancy by running multiple instances of each microservice to handle failovers.
    - **Why**: High availability and fault tolerance are critical for ensuring that the system remains operational even in the event of failures.
2. **Portability**:
    - **Requirement**: Ensure that the system can be deployed on any cloud provider.
    - **Implementation Details**:
        - Use Docker to containerize each microservice. Containers encapsulate all dependencies and environment settings, making the services portable across different environments.
        - Use Kubernetes for orchestration, as it is supported by all major cloud providers (e.g., AWS, GCP, Azure).
    - **Why**: Portability allows for flexibility in deployment, making it easier to switch cloud providers or run the system in hybrid cloud environments.
3. **Monitoring and Logging**:
    - **Requirement**: Implement comprehensive monitoring and logging to track performance and troubleshoot issues.
    - **Implementation Details**:
        - Use Prometheus to scrape and collect metrics from the microservices and Kubernetes nodes. Monitor CPU usage, memory usage, request count, and latency.
        - Use Grafana to visualize metrics and create dashboards for real-time monitoring.
        - Implement centralized logging to collect logs from all microservice instances. Tools like Elasticsearch, Logstash, and Kibana (ELK stack) can be used for log aggregation and analysis.
    - **Why**: Monitoring and logging are essential for maintaining visibility into the system’s performance and health. They help in diagnosing issues, optimizing performance, and ensuring reliability.

## Summary of Requirements and Architecture

- **Asynchronous Request Handling**: Use asynchronous web frameworks to handle I/O operations in a non-blocking manner, allowing multiple requests to be processed concurrently.
- **Microservices Architecture**: Separate the system into pre-processing, inference, and post-processing services, each responsible for specific tasks, promoting modularity and ease of management.
- **Scalability**: Implement horizontal scaling using Kubernetes and Horizontal Pod Autoscaler to handle increased load by adding more instances of microservices.
- **Load Balancing**: Use NGINX for external load balancing and an API Gateway for internal routing. Each microservice instance should run multiple worker processes to handle concurrent requests.
- **Performance**: Minimize latency through asynchronous processing and proactive scaling. Monitor custom metrics to ensure timely scaling.
- **Security**: Secure API endpoints with OAuth2 or OpenID Connect and enforce security policies through the API Gateway.
- **Multiple Workers**: Configure each microservice instance to run multiple workers, allowing efficient handling of concurrent requests within each instance.
- **Reliability**: Ensure high availability and fault tolerance with Kubernetes health checks and automatic failover mechanisms.
- **Portability**: Use Docker for containerization and Kubernetes for orchestration to ensure the system can be deployed on any cloud provider.
- **Monitoring and Logging**: Implement comprehensive monitoring with Prometheus and Grafana, and centralized logging with the ELK stack.

By adhering to these detailed functional and non-functional requirements, the system can efficiently handle high loads, maintain performance, and provide reliable service for customer churn predictions in a telecom context. Each requirement is linked to specific implementation strategies and technologies that fulfill the intended goals, ensuring a robust and scalable architecture.