
import xgboost as xgb
import numpy as np
import pandas as pd
import hypertune
import io
import argparse
from google.cloud import storage
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    accuracy_score, f1_score, roc_auc_score, average_precision_score
)
import time


# Function to load npz file from Google Cloud Storag
def load_npz_from_gcs(bucket_name, file_path):
    """
    Load a numpy NpzFile object from a file in Google Cloud Storage.

    Parameters:
    - bucket_name: str. The name of the GCS bucket where the file is stored.
    - file_path: str. The path to the file in the GCS bucket.

    Returns:
    - np.load(data_file): NpzFile. The data loaded from the file.
    """
    storage_client = storage.Client()  # Create a client to access the Google Cloud Storage service.
    bucket = storage_client.get_bucket(bucket_name)  # Create a bucket object for the specified bucket.
    blob = bucket.blob(file_path)  # Create a blob object for the specified file path.
    data_bytes = blob.download_as_bytes()  # Download the contents of the blob into memory.
    data_file = io.BytesIO(data_bytes)  # Use BytesIO to create a file-like object from the bytes data, which numpy can read from.
    return np.load(data_file)   # return the data as a numpy NpzFile object.


# Function to upload a file to Google Cloud Storage
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """
    Upload a file to Google Cloud Storage.

    Parameters:
    - bucket_name: str. The name of the GCS bucket where the file will be stored.
    - source_file_name: str. The path to the file on the local system.
    - destination_blob_name: str. The path where the file will be stored in the GCS bucket.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


# Function to parse command line arguments
def parse_args():
    """
    Parse command line arguments.

    Returns:
    - args: Namespace. The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', help='Name of the GCS bucket', required=True)
    parser.add_argument('--file_path', help='Path to data', required=True)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--lambda_value', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    return parser.parse_args()


# Function to manage model metadata
def generate_model_metadata(timestamp, y_true, y_pred, y_pred_proba=None, hyperparameters=None):
    """
    Generate metadata for a model, including various performance metrics and hyperparameters.

    Parameters:
    - timestamp: str. Unix timestamp.
    - y_true: array-like. True labels.
    - y_pred: array-like. Predicted labels.
    - y_pred_proba: array-like (default None). Predicted probabilities for the positive class.
    - hyperparameters: dict (default None). Hyperparameters of the model.

    Returns:
    - metadata: dict. A dictionary with metric names as keys and their corresponding values, as well as the model's hyperparameters.
    """
    metadata = {
        'Unix Time': timestamp,
        'Confusion Matrix': confusion_matrix(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else 'N/A',
        'AUC-PR': average_precision_score(y_true, y_pred_proba) if y_pred_proba is not None else 'N/A',
    }

    if hyperparameters is not None:
        metadata.update(hyperparameters)

    return metadata


# Function to print the evaluation metrics
def print_evaluation(metrics):
    """
    Prints the evaluation metrics from the evaluate_model function.

    Parameters:
    - metrics: dict. The metrics to print.
    """
    for key, value in metrics.items():
        if key != 'Model':
            print(f"{key}: {value}")
        else:
            print(f"{value} Evaluation")
    print("\n")  # New line for better readability between model evaluations


# Main function
def main():
    """
    Main function to run the script.
    """
    args = parse_args()
    data = load_npz_from_gcs(args.bucket_name, args.file_path)

    # Extract the arrays for training, validation, and testing
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    num_negative = np.sum(y_train == 0)
    num_positive = np.sum(y_train == 1)
    scale_pos_weight = num_negative / num_positive

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'lambda': args.lambda_value,
        'alpha': args.alpha,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'scale_pos_weight': scale_pos_weight,
    }

    epochs = 300  # Number of epochs

    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, epochs, evals, early_stopping_rounds=5)
    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)

    unix_timestamp = time.time()
    unix_timestamp_str = str(unix_timestamp).replace('.', '')

    # Evaluate the model
    xgb_metadata = generate_model_metadata(unix_timestamp, y_test, y_pred_binary, y_pred, params)
    print_evaluation(xgb_metadata)

    # Report the RMSE to hyperparameter tuning service
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='acupr',
        metric_value=xgb_metadata['AUC-PR'],
        global_step=epochs
    )

    # Assuming metrics have been collected for all models into a list of dictionaries
    all_metrics = [xgb_metadata]

    # Convert to DataFrame for easy viewing
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f'xgb_metadata{unix_timestamp_str}.csv', index=False)

    # After training and saving the model...
    model.save_model(f'xgboost_model{unix_timestamp_str}.json')

    # GCS path for the model and Metadata
    gcs_model_path = f'model/xgboost_model{unix_timestamp_str}.json'
    gcs_metrices_path = f'metadata/xgb_metadata{unix_timestamp_str}.csv'

    # Upload the model and Metadata to GCS
    upload_to_gcs(args.bucket_name, f'xgboost_model{unix_timestamp_str}.json', gcs_model_path)
    upload_to_gcs(args.bucket_name, f'xgb_metadata{unix_timestamp_str}.csv', gcs_metrices_path)


if __name__ == '__main__':
    main()
