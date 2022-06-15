import pandas as pd
import os
import logging
import pickle
import time

from typing import Tuple, Dict
from sklearn.linear_model import LogisticRegression
from google.cloud import storage


logging.basicConfig(level=logging.DEBUG)
COLUMNS_TO_DROP = ['ChestPainType', 'RestingECG', 'ST_Slope', 'Oldpeak', 'ExerciseAngina', 'FastingBS']

def load_data(path) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.drop(COLUMNS_TO_DROP, axis=1)

    df['RestingBP_bin'] = df['RestingBP'].apply(lambda x: 1 if x>120 else 0)
    df['Cholestrol_bin'] = df['Cholesterol'].apply(lambda x: 1 if x>200 else 0)

    df = df.drop(['RestingBP', 'Cholesterol'], axis=1)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(["HeartDisease"], axis=1)
    y = df["HeartDisease"]

    return X, y

def upload_model(model: LogisticRegression, bucket:str) -> None:
    artifact_filename = f'{time.time_ns()}_model.pkl'
    local_path = artifact_filename
    with open(local_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    storage_path = os.path.join(bucket, artifact_filename)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(local_path)

    os.remove(local_path)

def train(hparams: Dict[str, str]) -> None:
    model = LogisticRegression(penalty=hparams["regularization"], solver='saga')
    logging.info(f"Training model with regularization: {hparams['regularization']}")
    data = load_data(hparams["input_path"])
    X, y = preprocess_data(data)
    model.fit(X, y)
    logging.info(f"Linear regression model acieved training accuracy of {model.score(X, y)}")

    upload_model(model, hparams["output_path"])
