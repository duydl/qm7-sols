from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def get_models(model_names=None):
    standard_scaling = [('scaler', StandardScaler())]
    pca = [('pca', PCA(n_components=0.95))]

    all_models = {
        'Linear': {
            'model': Pipeline(standard_scaling + [('regressor', LinearRegression())]),
            'params': {}
        },
        'Ridge': {
            'model': Pipeline(standard_scaling + [('regressor', Ridge())]),
            'params': {
                'regressor__alpha': np.linspace(0, 10, 11)
            }
        },
        'KernelRidge_NoScaling': {
            'model': Pipeline(
                # standard_scaling +
                [('regressor', KernelRidge(kernel='rbf'))]),
            'params': {
                'regressor__alpha': np.logspace(-5, -3, 3),
                'regressor__gamma': np.logspace(-5, -3, 3),
            }
        },
        'KernelRidge': {
            'model': Pipeline(
                standard_scaling +
                [('regressor', KernelRidge(kernel='rbf'))]),
            'params': {
                # 'regressor__alpha': np.logspace(-5, -3, 3),
                # 'regressor__gamma': np.logspace(-5, -3, 3),
                # 'regressor__alpha': np.logspace(-4.5, -3.5, 3),
                # 'regressor__gamma': np.logspace(-3, -1, 3),
                'regressor__alpha': np.linspace(0.0004, 0.0004, 1),
                'regressor__gamma': np.linspace(0.01, 0.01, 1),
            }
        },
        'SVR': {
            'model': Pipeline(
                standard_scaling + 
                [('regressor', SVR(kernel='rbf'))]),
            'params': {
                'regressor__C': np.logspace(3, 5, 3),
                'regressor__epsilon': np.logspace(0, 3, 3),
                # 'regressor__gamma': np.logspace(-5, -4, 2),
            }
        },
        'KNN': {
            'model': Pipeline(
                standard_scaling +
                [('regressor', KNeighborsRegressor())]),
            'params': {
                'regressor__n_neighbors': np.linspace(1, 10, 5).astype(int),
                'regressor__weights': ['uniform', 'distance'],
                'regressor__p': np.array([1, 2])  # 1 for Manhattan distance, 2 for Euclidean distance
            }
        }
    }

    if model_names is None:
        return all_models

    return {name: all_models[name] for name in model_names if name in all_models}

