# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import gzip
import json
import pickle
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def clean_data(df):
    df = df.copy()

    if "default payment next month" in df.columns:
        df.rename(
            columns={"default payment next month": "default"},
            inplace=True,
        )

    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace(
            [0, 5, 6],
            4,
        )

    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace(
            0,
            3,
        )

    return df


def build_pipeline(X):

    categorical_features = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]

    numeric_features = [
        col
        for col in X.columns
        if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif)),
            ("classifier", SVC()),
        ]
    )

    return pipeline


def metrics_dictionary(y_true, y_pred, dataset):
    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": round(
            precision_score(y_true, y_pred),
            3,
        ),
        "balanced_accuracy": round(
            balanced_accuracy_score(y_true, y_pred),
            3,
        ),
        "recall": round(
            recall_score(y_true, y_pred),
            3,
        ),
        "f1_score": round(
            f1_score(y_true, y_pred),
            3,
        ),
    }


def confusion_matrix_dictionary(y_true, y_pred, dataset):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(tn),
            "predicted_1": int(fp),
        },
        "true_1": {
            "predicted_0": int(fn),
            "predicted_1": int(tp),
        },
    }


def save_model(model):

    Path("files/models").mkdir(
        parents=True,
        exist_ok=True,
    )

    with gzip.open(
        "files/models/model.pkl.gz",
        "wb",
    ) as file:
        pickle.dump(model, file)


def save_metrics(results):

    Path("files/output").mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        "files/output/metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:
            file.write(json.dumps(result))
            file.write("\n")


def main():

    train = pd.read_csv(
        "files/input/train_data.csv.zip"
    )

    test = pd.read_csv(
        "files/input/test_data.csv.zip"
    )

    train = clean_data(train)
    test = clean_data(test)

    X_train = train.drop(
        columns=["default"]
    )

    y_train = train["default"]

    X_test = test.drop(
        columns=["default"]
    )

    y_test = test["default"]

    pipeline = build_pipeline(X_train)

    param_grid = {
        "selector__k": [
            10
        ],
        "classifier__kernel": [
            "linear"
        ],
        "classifier__C": [
            10
        ]
    }

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=3,
        n_jobs=-1,
        refit=True,
        verbose=3,
    )
    print("Iniciando GridSearch")
    print(X_train.shape)
    model.fit(
        X_train,
        y_train,
    )
    print("GridSearch terminado")

    print(model.best_params_)
    print(model.best_score_)
    print(
        "train score:",
        model.score(X_train, y_train),
    )
    print(
        "test score :",
        model.score(X_test, y_test),
    )

    y_train_pred = model.predict(
        X_train
    )

    y_test_pred = model.predict(
        X_test
    )

    results = [
        metrics_dictionary(
            y_train,
            y_train_pred,
            "train",
        ),
        metrics_dictionary(
            y_test,
            y_test_pred,
            "test",
        ),
        confusion_matrix_dictionary(
            y_train,
            y_train_pred,
            "train",
        ),
        confusion_matrix_dictionary(
            y_test,
            y_test_pred,
            "test",
        ),
    ]

    save_model(model)
    save_metrics(results)


if __name__ == "__main__":
    main()