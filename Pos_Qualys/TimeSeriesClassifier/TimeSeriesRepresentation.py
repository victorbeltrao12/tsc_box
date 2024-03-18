import numpy as np
import pywt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation, SymbolicAggregateApproximation
from scipy.fftpack import fft

def choose_wavelet(X, candidate_wavelets):
    min_variance = float('inf')
    best_wavelet = None
    
    for wavelet_type in candidate_wavelets:
        coeffs_cA, coeffs_cD = pywt.dwt(X, wavelet_type, axis=1)
        total_variance = np.var(coeffs_cA) + np.var(coeffs_cD)
        
        if total_variance < min_variance:
            min_variance = total_variance
            best_wavelet = wavelet_type
            
    return best_wavelet

def transform_data_math(X):
    n_sax_symbols = int(X.shape[1] / 4)
    n_paa_segments = int(X.shape[1] / 4)

    X_fft = np.abs(fft(X, axis=1))

    # Escolha automaticamente a melhor wavelet
    candidate_wavelets = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9']
    best_wavelet = choose_wavelet(X, candidate_wavelets)
    
    coeffs_cA, coeffs_cD = pywt.dwt(X, best_wavelet, axis=1)
    X_dwt = np.hstack((coeffs_cA, coeffs_cD))

    paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
    X_paa_ = paa.inverse_transform(paa.fit_transform(X))
    X_paa = X_paa_.reshape(X_paa_.shape[0], -1)
    stats_PAA = np.hstack([np.mean(X_paa, axis=1).reshape(-1,1),
                           np.std(X_paa, axis=1).reshape(-1,1),
                           np.max(X_paa, axis=1).reshape(-1,1),
                           np.min(X_paa, axis=1).reshape(-1,1),
                           ])

    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    X_sax_ = sax.inverse_transform(sax.fit_transform(X))
    X_sax = X_sax_.reshape(X_sax_.shape[0], -1)
    stats_SAX = np.hstack([np.mean(X_sax, axis=1).reshape(-1,1),
                           np.std(X_sax, axis=1).reshape(-1,1),
                           np.max(X_sax, axis=1).reshape(-1,1),
                           np.min(X_sax, axis=1).reshape(-1,1),
                           ])

    data_X = TimeSeriesScalerMeanVariance().fit_transform(X)
    data_X.resize(data_X.shape[0], data_X.shape[1])
    stats_X = np.hstack([np.mean(data_X, axis=1).reshape(-1,1),
                         np.std(data_X, axis=1).reshape(-1,1),
                         np.max(data_X, axis=1).reshape(-1,1),
                         np.min(data_X, axis=1).reshape(-1,1),
                         ])

    data_FFT = TimeSeriesScalerMeanVariance().fit_transform(X_fft)
    data_FFT.resize(data_FFT.shape[0], data_FFT.shape[1])
    stats_FFT = np.hstack([np.mean(data_FFT, axis=1).reshape(-1,1),
                           np.std(data_FFT, axis=1).reshape(-1,1),
                           np.max(data_FFT, axis=1).reshape(-1,1),
                           np.min(data_FFT, axis=1).reshape(-1,1),
                           ])

    data_DWT = TimeSeriesScalerMeanVariance().fit_transform(X_dwt)
    data_DWT.resize(data_DWT.shape[0], data_DWT.shape[1])
    stats_DWT = np.hstack([np.mean(data_DWT, axis=1).reshape(-1,1),
                           np.std(data_DWT, axis=1).reshape(-1,1),
                           np.max(data_DWT, axis=1).reshape(-1,1),
                           np.min(data_DWT, axis=1).reshape(-1,1),
                           ])

    return {
        "TS": np.hstack([data_X, stats_X]),
        "FFT": np.hstack([data_FFT, stats_FFT]),
        "DWT": np.hstack([data_DWT, stats_DWT]),
        "PAA": np.hstack([X_paa, stats_PAA]),
        "SAX": np.hstack([X_sax, stats_SAX])
    }

def transform_data(X):
    n_sax_symbols = int(X.shape[1] / 4)
    n_paa_segments = int(X.shape[1] / 4)

    X_fft = np.abs(fft(X, axis=1))

    candidate_wavelets = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9']
    best_wavelet = choose_wavelet(X, candidate_wavelets)
    
    coeffs_cA, coeffs_cD = pywt.dwt(X, best_wavelet, axis=1)
    X_dwt = np.hstack((coeffs_cA, coeffs_cD))

    paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
    X_paa_ = paa.inverse_transform(paa.fit_transform(X))
    X_paa = X_paa_.reshape(X_paa_.shape[0], -1)

    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    X_sax_ = sax.inverse_transform(sax.fit_transform(X))
    X_sax = X_sax_.reshape(X_sax_.shape[0], -1)

    data = np.concatenate((X, X_fft, X_dwt, X_paa, X_sax), axis=1)
    data_concat = TimeSeriesScalerMeanVariance().fit_transform(data)
    data_concat.resize(data.shape[0], data.shape[1])
    
    return data_concat