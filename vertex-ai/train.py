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


def load_npz_from_gcs(bucket_name, file_path):
    storage_client = storage.Client()  # Create a client to access the Google Cloud Storage service.
    bucket = storage_client.get_bucket(bucket_name)  # Create a bucket object for the specified bucket.
    blob = bucket.blob(file_path)  # Create a blob object for the specified file path.
    data_bytes = blob.download_as_bytes()  # Download the contents of the blob into memory.
    data_file = io.BytesIO(data_bytes)  # Use BytesIO to create a file-like object from the bytes data, which numpy can read from.
    return np.load(data_file)   # return the data as a numpy NpzFile object.


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def parse_args():
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


def evaluate_model(timestamp, y_true, y_pred, y_pred_proba=None, params=None):
    """
    Evaluate a model on various performance metrics.

    Parameters:
    - model_name: str. Name of the model being evaluated.
    - y_true: array-like. True labels.
    - y_pred: array-like. Predicted labels.
    - y_pred_proba: array-like (default None). Predicted probabilities for the positive class.
    - params: dict (default None). Hyperparameters of the model.

    Returns:
    - A dictionary with metric names as keys and their corresponding values.
    """
    metrics = {
        'Unix Time': timestamp,
        'Confusion Matrix': confusion_matrix(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else 'N/A',
        'AUC-PR': average_precision_score(y_true, y_pred_proba) if y_pred_proba is not None else 'N/A',
        'max_depth': params['max_depth'],
        'eta': params['eta'],
        'lambda': params['lambda'],
        'alpha': params['alpha'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree']
    }

    return metrics


def print_evaluation(metrics):
    """
    Prints the evaluation metrics from the evaluate_model function.
    """
    for key, value in metrics.items():
        if key != 'Model':
            print(f"{key}: {value}")
        else:
            print(f"{value} Evaluation")
    print("\n")  # New line for better readability between model evaluations


def main():
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
    xgb_metadata = evaluate_model(unix_timestamp, y_test, y_pred_binary, y_pred, params)
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

    # Specify your desired GCS path for the model
    gcs_model_path = f'model/xgboost_model{unix_timestamp_str}.json'
    gcs_metrices_path = f'metadata/xgb_metadata{unix_timestamp_str}.csv'

    # Upload the model to GCS
    upload_to_gcs(args.bucket_name, f'xgboost_model{unix_timestamp_str}.json', gcs_model_path)
    upload_to_gcs(args.bucket_name, f'xgb_metadata{unix_timestamp_str}.csv', gcs_metrices_path)


if __name__ == '__main__':
    main()
