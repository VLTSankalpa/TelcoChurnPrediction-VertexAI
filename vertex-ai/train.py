import xgboost as xgb
import hypertune
import pandas as pd
import numpy as np
import io
import argparse
from google.cloud import storage
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


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
        'eval_metric': 'logloss'
    }

    epochs = 300  # Number of epochs

    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, epochs, evals, early_stopping_rounds=5)
    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)
    # Evaluate the model
    conf_mat = confusion_matrix(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    accuracy = accuracy_score(y_test, y_pred_binary)

    print('Confusion Matrix:\n', conf_mat)
    print(f'Precision: {precision:.5f}')
    print(f'Recall: {recall:.5f}')
    print(f'Accuracy: {accuracy:.5f}')

    # Report the RMSE to hyperparameter tuning service
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=accuracy,
        global_step=epochs
    )

    # After training and saving the model...
    model.save_model('xgboost_model.json')

    # Specify your desired GCS path for the model
    gcs_model_path = 'model/xgboost_model.json'

    # Upload the model to GCS
    upload_to_gcs(args.bucket_name, 'xgboost_model.json', gcs_model_path)


if __name__ == '__main__':
    main()
