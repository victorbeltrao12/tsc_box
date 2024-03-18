import pandas as pd
import numpy as np
from aeon.datasets import load_classification
from aeon.datasets.tsc_data_lists import univariate_equal_length
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier, ShapeDTW, ElasticEnsemble
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation, SymbolicAggregateApproximation
import pywt
from sklearn.model_selection import LeaveOneOut
from slearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.fftpack import fft
from numba import jit
from tqdm import tqdm
import timeit
from datetime import timedelta

#own library
from TimeSeriesClassifier.TimeSeriesRepresentation import (
    transform_data, transform_data_math
)

import warnings
warnings.filterwarnings("ignore")

class TimeSeriesClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.trained_models = {}
        """_summary_

        Returns:
            _type_: _description_
        """
    
    @staticmethod
    def select_model(option, random_state):
        if option == '1nn':
            return KNeighborsTimeSeriesClassifier(distance='euclidean', n_neighbors=1, n_jobs=-1)
        elif option == '3nn':
            return KNeighborsTimeSeriesClassifier(distance='dtw', n_neighbors=3, n_jobs=-1)
        elif option == 'svm':
            return SVC(C = 100, gamma=0.01, kernel='linear', probability=True)
        elif option == 'gbc':
            return GradientBoostingClassifier(n_estimators=5, random_state=random_state)
        elif option == 'nb':
            return GaussianNB()
        elif option == 'shape':
            return ShapeDTW(n_neighbors=1)
        elif option == 'ee':
            return ElasticEnsemble(n_jobs=-1, random_state=random_state, majority_vote=True)
        elif option == 'exrf':
            return ExtraTreesClassifier(n_estimators=200, criterion="entropy", bootstrap=True, max_features="sqrt", oob_score=True, n_jobs=-1, random_state=None)
        elif option == 'rd':
            return RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        else:
            return RandomForestClassifier(n_estimators=200, criterion="gini", max_features="sqrt", n_jobs=-1, random_state=None)
    
    @jit
    def train_with_meta_classifier(X_train, y_train, base_option='exrf', meta_option='rd', random_state=42, method=None):
        if method is None: 
            X_train_transformed = transform_data(X_train)
        else:
            X_train_transformed = transform_data_math(X_train)

        loo = LeaveOneOut()
        loo.get_n_splits(X_train_transformed)

        # Treinar um modelo para todos os dados transformados
        model = TimeSeriesClassifier.select_model(base_option, random_state)
        for train_index, test_index in tqdm(loo.split(X_train_transformed), colour='red', desc="Training"):
            X_train_fold, _ = X_train_transformed[train_index], X_train_transformed[test_index]
            y_train_fold, _ = y_train[train_index], y_train[test_index]
            model.fit(X_train_fold, y_train_fold)

        # Preparar dados para o meta-classificador
        meta_features = []
        for X_trans in X_train_transformed:
            instance_features = []
            proba = model.predict_proba(X_trans.reshape(1, -1)) # Reshape para compatibilidade com predict_proba
            proba /= np.sum(proba)
            instance_features.extend(proba.flatten())
            meta_features.append(instance_features)

        meta_features = np.array(meta_features)

        # Treinar o meta-classificador
        meta_classifier = TimeSeriesClassifier.select_model(meta_option, random_state=random_state)
        meta_classifier.fit(meta_features, y_train)

        return model, meta_classifier

    @jit
    def predict_with_meta_classifier(X_test, trained_base_models, trained_meta_classifier):
        predictions = []
        meta_features_test = []  # Inicialize uma lista para armazenar todos os meta-recursos dos dados de teste
    
        for i in tqdm(range(len(X_test)), ascii=True, colour='green', desc="Predict"):
            x_instance = X_test[i].reshape(1, -1)
            x_transformed = TimeSeriesClassifier.transform_data_math(x_instance)
    
            instance_features = np.zeros(trained_meta_classifier.n_features_in_)  # Inicialize um vetor para armazenar as características do meta-classificador
            for rep, model in trained_base_models.items():
                proba = model.predict_proba(x_transformed[rep][0].reshape(1, -1))
                instance_features += proba.flatten()  # Adicione as probabilidades para cada classe
    
            meta_feature = np.array(instance_features).reshape(1, -1)
            predictions.append(trained_meta_classifier.predict(meta_feature)[0])  # Adicionar a previsão à lista de previsões
    
            meta_features_test.append(meta_feature.flatten())  # Adicionar meta-recursos da instância atual à lista
    
        # Converter a lista de meta-recursos dos dados de teste em um array numpy
        meta_features_test = np.array(meta_features_test)
            
        return predictions