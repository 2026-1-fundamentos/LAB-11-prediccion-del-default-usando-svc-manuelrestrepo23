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
import os
import pickle
import zipfile

import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.feature_selection import SelectKBest, f_classif  # type: ignore
from sklearn.metrics import (  # type: ignore
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore


def cargar_datos(ruta_zip):

    with zipfile.ZipFile(ruta_zip) as archivo_zip:
        nombre_csv = archivo_zip.namelist()[0]
        with archivo_zip.open(nombre_csv) as archivo_csv:
            df = pd.read_csv(archivo_csv)
    return df


def limpiar_datos(df):

    df = df.copy()

    df = df.rename(columns={"default payment next month": "default"})

    df = df.drop(columns=["ID"])

    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"] <= 4, 4)

    df = df.dropna().reset_index(drop=True)

    return df

def dividir_features_target(df):
    """Divide el DataFrame en features y target"""

    x = df.drop(columns=["default"])
    y = df["default"].astype(int)
    return x, y

def construir_y_optimizar_pipeline(x_train, y_train):

    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    columnas_numericas = [c for c in x_train.columns if c not in columnas_categoricas]

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), columnas_categoricas),
            ("scaler", StandardScaler(), columnas_numericas),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("pca", PCA()),
            ("selector", SelectKBest(score_func=f_classif)),
            ("clasificador", SVC(kernel="rbf", random_state=12345, max_iter=-1)),
        ]
    )

    parametros = {
        "pca__n_components": [20, x_train.shape[1] - 2],
        "selector__k": [12],
        "clasificador__kernel": ["rbf"],
        "clasificador__gamma": [0.1],
    }

    modelo = GridSearchCV(
        pipeline,
        parametros,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=1,
        refit=True,
    )

    modelo.fit(x_train, y_train)

    return modelo


def guardar_modelo(modelo, ruta_salida):
    """Guarda el modelo comprimido con gzip"""

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    with gzip.open(ruta_salida, "wb") as archivo:
        pickle.dump(modelo, archivo)


def calcular_metricas(modelo, x_train, y_train, x_test, y_test):

    metricas = []
    matrices = []

    mejor_estimador = modelo.best_estimator_

    for nombre_dataset, x, y in [
        ("train", x_train, y_train),
        ("test", x_test, y_test),
    ]:
        predicciones = mejor_estimador.predict(x)

        metricas.append(
            {
                "type": "metrics",
                "dataset": nombre_dataset,
                "precision": float(precision_score(y, predicciones, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y, predicciones)),
                "recall": float(recall_score(y, predicciones, zero_division=0)),
                "f1_score": float(f1_score(y, predicciones, zero_division=0)),
            }
        )

        cm = confusion_matrix(y, predicciones, labels=[0, 1])
        matrices.append(
            {
                "type": "cm_matrix",
                "dataset": nombre_dataset,
                "true_0": {
                    "predicted_0": int(cm[0][0]),
                    "predicted_1": int(cm[0][1]),
                },
                "true_1": {
                    "predicted_0": int(cm[1][0]),
                    "predicted_1": int(cm[1][1]),
                },
            }
        )

    return metricas + matrices


def guardar_metricas(metricas, ruta_salida):

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        for metrica in metricas:
            archivo.write(json.dumps(metrica) + "\n")


def main():

    train_df = cargar_datos("files/input/train_data.csv.zip")
    test_df = cargar_datos("files/input/test_data.csv.zip")

    train_df = limpiar_datos(train_df)
    test_df = limpiar_datos(test_df)

    x_train, y_train = dividir_features_target(train_df)
    x_test, y_test = dividir_features_target(test_df)

    modelo = construir_y_optimizar_pipeline(x_train, y_train)

    guardar_modelo(modelo, "files/models/model.pkl.gz")

    metricas = calcular_metricas(modelo, x_train, y_train, x_test, y_test)
    guardar_metricas(metricas, "files/output/metrics.json")


if __name__ == "__main__":
    main()


