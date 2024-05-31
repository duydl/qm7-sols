import json, os, scipy
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from models import get_models


def load_and_preprocess_data(filepath, fold=1, feature="eigen"):
    dataset = scipy.io.loadmat(filepath)
    y = dataset["T"].squeeze(0)
    
    if feature == "eigen":
        eigenvalues = np.linalg.eigvalsh(dataset["X"])
        sorted_eigenvalues = np.sort(np.abs(eigenvalues), axis=1)[:, ::-1]
        X = sorted_eigenvalues
        return X, y


def evaluate_models(X, y, model_names=None, verbose=2):
    models = get_models(model_names=model_names)
    results_file_path = "logs/grid_search_results.json"
    
    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as file:
            results = json.load(file)
    else:
        results = {}
        
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=10)
    
    for model_name, model_info in models.items():
        grid_search = GridSearchCV(
            model_info['model'], 
            model_info['params'], 
            cv=inner_cv, 
            scoring='neg_mean_squared_error', 
            verbose=verbose,
        )
        grid_check = grid_search.fit(X, y)

        results[model_name] = []
        for mean_score, params in zip(
            grid_check.cv_results_['mean_test_score'], 
            grid_check.cv_results_['params']
        ):
            results[model_name].append({
                'MSE': -mean_score,
                'parameters': params
            })
        
        best_model, best_score = grid_check.best_estimator_, -grid_check.best_score_
        print(f"Best model for {model_name}: {best_model}, Best MSE: {best_score}")
    
    with open("logs/grid_search_results.json", "w") as file:
        json.dump(results, file, indent=4)


def run_best_model(model, X, y, model_name):
    pass

filepath = "../../data/raw/qm7.mat"
X, y = load_and_preprocess_data(filepath)
evaluate_models(X, y, model_names=["KernelRidge"])