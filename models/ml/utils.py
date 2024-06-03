import json, os, scipy
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from models import get_models


def load_data(filepath, feature="eigenspectrum"):
    dataset = scipy.io.loadmat(filepath)
    y = atom_es = dataset["T"].squeeze()
    coulomb_matrices = dataset["X"]
    
    if feature == "eigenspectrum":
        # Compute eigenvalues of the matrices and sort them by magnitude in descending order
        X = get_sorted_eigenvals(coulomb_matrices)
        return X, y

    elif feature == "sorted_coulomb":
        # Sort Coulomb matrices by the norm-2 of their columns in descending order
        X = sort_coulomb_matrics(coulomb_matrices)
        return X, y
    
    elif feature == "expanded_coulomb":
        X, y = expand_coulomb_matrices(coulomb_matrices)
        return X, y
    else:
        raise ValueError("Feature type not supported")

def get_sorted_eigenvals(coulomb_matrices):
    eigenvals = np.linalg.eigvalsh(coulomb_matrices)
    sorted_eigenvals = np.sort(np.abs(eigenvals), axis=1)[:, ::-1]
    return sorted_eigenvals

def sort_coulomb_matrics(coulomb_matrices):
    sorted_coulomb_matrices = np.array([
            cm[:, np.argsort(-np.linalg.norm(cm, axis=0))] for cm in coulomb_matrices
        ])
    return sorted_coulomb_matrices

def expand_coulomb_matrices(coulomb_matrices, atom_es, noise_level=1.0, n_expanded_samples=100):
    random_coulomb_matrices = []
    new_atom_es = []
    
    for coulomb_mat, e in zip(coulomb_matrices, atom_es):
        row_norms = np.linalg.norm(coulomb_mat, axis=1)

        for _ in range(n_expanded_samples):
            noise = np.random.normal(0, noise_level, size=row_norms.shape)
            permutation = np.argsort(-(row_norms + noise))
            augmented_coulomb = coulomb_mat[permutation, :][:, permutation]
            random_coulomb_matrices.append(augmented_coulomb)
        
        new_atom_es.extend([e] * n_expanded_samples)

    return random_coulomb_matrices, new_atom_es


def evaluate_models(X, y, model_names=None, verbose=2):
    models = get_models(model_names=model_names)
    results_file_path = "logs/grid_search_results.json"
    
    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as file:
            results = json.load(file)
    else:
        results = {}
        
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=10)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    for model_name, model_info in models.items():
        if model_name not in results:
            results[model_name] = []
            
        grid_search = GridSearchCV(
            model_info['model'], 
            model_info['params'], 
            cv=inner_cv,
            scoring='neg_mean_squared_error',
            n_jobs=5,
            verbose=verbose,
        )
        grid_check = grid_search.fit(X, y)

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
X, y = load_data(filepath)
evaluate_models(X, y, model_names=["Linear"])