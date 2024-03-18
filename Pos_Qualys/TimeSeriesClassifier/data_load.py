import pandas as pd
import numpy as np
from aeon.datasets import load_classification
from aeon.datasets.tsc_data_lists import univariate_equal_length
from slearn.preprocessing import LabelEncoder
import timeit
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from TimeSeriesClassifier.TimeSeriesClassifier import(
    select_model, train_with_meta_classifier, predict_with_meta_classifier
)

from TimeSeriesClassifier.TimeSeriesRepresentation import(
    transform_data, transform_data_math
)

@staticmethod
def load_data(dataset):
    # LabelEncoder para labels alvo
    le = LabelEncoder()
    
    # Carregar conjunto de dados do repositório UCR
    X_train, y_train = load_classification(dataset, split="TRAIN")
    X_test, y_test = load_classification(dataset, split="test")
    
    # Formatar o conjunto de dados para 2D
    features_train = X_train.reshape(X_train.shape[0], -1)
    features_test = X_test.reshape(X_test.shape[0], -1)
    
    # Ajustar e transformar as labels alvo
    target_train = le.fit_transform(y_train)
    target_test = le.transform(y_test)
    
    return features_train, features_test, target_train, target_test

@staticmethod
def evaluate_datasets(list_datasets):
    le = LabelEncoder()
    accuracy_data = []
    total_time = 0
    for dataset_name in list_datasets:
        X_train, y_train = load_classification(dataset, split="TRAIN")
        X_test, y_test = load_classification(dataset, split="test")
        # Formatar o conjunto de dados para 2D
        features_train = X_train.reshape(X_train.shape[0], -1)
        features_test = X_test.reshape(X_test.shape[0], -1)
        # Ajustar e transformar as labels alvo
        target_train = le.fit_transform(y_train)
        target_test = le.transform(y_test)

        dataset_accuracies = []
        start = timeit.default_timer()
        base_models, meta_classifier = TimeSeriesClassifier.train_with_meta_classifier(features_train, target_train, base_option='exrf', meta_option='rd')
        predictions = TimeSeriesClassifier.predict_with_meta_classifier(features_test, base_models, meta_classifier)
        stop = timeit.default_timer()
        total_time += stop - start
        accuracy = np.mean(predictions == target_test)
        dataset_accuracies.append(accuracy)

        print(f"Accuracy {dataset_name}: {accuracy}, total time: {timedelta(total_time)}")
        accuracy_data.append({'Dataset Name': dataset_name, 'Accuracy': accuracy})

    accuracy_df = pd.DataFrame(accuracy_data)
    return accuracy_df

@staticmethod
def train_predict(features_train, features_test, target_train, target_test, base_models=None, meta_classifier=None):
    if base_models is None or meta_classifier is None:
        # Treinar com base em uma série temporal e um classificador meta
        base_models, meta_classifier = TimeSeriesClassifier.train_with_meta_classifier(features_train, target_train, base_option='exrf', meta_option='rd')
    
    # Prever com base nos modelos treinados
    predictions = TimeSeriesClassifier.predict_with_meta_classifier(features_test, base_models, meta_classifier)
    
    # Calcular a precisão
    accuracy = np.mean(predictions == target_test)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    # Dataset único para teste
    dataset = "Coffee"
    
    # Carregar dados
    features_train, features_test, target_train, target_test = load_data(dataset)
    
    # Treinar e prever
    train_predict(features_train, features_test, target_train, target_test)
