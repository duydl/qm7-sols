from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def get_models(model_names=None):
    preprocessing_steps = [('scaler', StandardScaler())]

    all_models = {
        'Linear': {
            'model': Pipeline(preprocessing_steps + [('regressor', LinearRegression())]),
            'params': {}
        },
        'Ridge': {
            'model': Pipeline(preprocessing_steps + [('regressor', Ridge())]),
            'params': {
                # 'regressor__alpha': np.logspace(-5, 3, 3),
                'regressor__alpha': np.linspace(0, 0.2, 11)
            }
        },
        'KernelRidge': {
            'model': Pipeline(preprocessing_steps + [('regressor', KernelRidge())]),
            'params': {
                'regressor__alpha': np.logspace(-5, -3, 3),
                'regressor__gamma': np.logspace(-5, -3, 3)
            }
        },
        'SVR': {
            'model': Pipeline(preprocessing_steps + [('regressor', SVR(kernel='rbf'))]),
            'params': {
                'regressor__C': np.logspace(4, 4, 1),
                'regressor__gamma': np.logspace(-5, -4, 2),
                'regressor__epsilon': np.logspace(0, 1, 2)
            }
        }
    }

    if model_names is None:
        return all_models

    return {name: all_models[name] for name in model_names if name in all_models}

