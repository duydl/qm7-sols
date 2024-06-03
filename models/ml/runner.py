import json
import os
import argparse
import scipy
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV,PredefinedSplit
from models import get_models

def numpy_to_python(obj):
    """Convert numpy objects to native Python objects to ensure JSON serialization compatibility."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def load_data(filepath, feature="eigenspectrum"):
    dataset = scipy.io.loadmat(filepath)
    y = atom_es = dataset["T"].squeeze()
    coulomb_matrices = dataset["X"]
    
    test_fold = np.empty(coulomb_matrices.shape[0], dtype=int)
    for i in range(5):
        test_fold[dataset["P"][i].flatten()] = i

    if feature == "eigenspectrum":
        X = get_sorted_eigenvals(coulomb_matrices)
        return X, y, test_fold

    elif feature == "sorted_coulomb":
        X = sort_coulomb_matrics(coulomb_matrices)
        return X, y, test_fold
    
    elif feature == "expanded_coulomb":
        X, y = expand_coulomb_matrices(coulomb_matrices, atom_es)
        return X, y, test_fold
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

    return np.array(random_coulomb_matrices), np.array(new_atom_es)

def evaluate_models(X, y, test_fold, model_names=None, verbose=2):
    models = get_models(model_names=model_names)
    results_file_path = "logs/grid_search_results.json"
    
    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as file:
            results = json.load(file)
    else:
        results = {}
        
    # inner_cv = KFold(n_splits=5, shuffle=True, random_state=10)
    inner_cv = PredefinedSplit(test_fold)
    
    for model_name, model_info in models.items():
        if model_name not in results:
            results[model_name] = []
            
        grid_search = GridSearchCV(
            model_info["model"], 
            model_info["params"], 
            cv=inner_cv,
            scoring="neg_mean_absolute_error",
            n_jobs=5,
            verbose=verbose,
        )
        grid_check = grid_search.fit(X, y)

        for mean_score, params in zip(
            grid_check.cv_results_["mean_test_score"], 
            grid_check.cv_results_["params"]
        ):
            params = {k: numpy_to_python(v) for k, v in params.items()}
            results[model_name].append({
                "MAE": -mean_score,
                "parameters": params
            })
       
        best_model, best_score = grid_check.best_estimator_, -grid_check.best_score_
        print(f"Best model for {model_name}: {best_model}, Best MAE: {best_score}")
    
    with open("logs/grid_search_results.json", "w") as file:
        json.dump(results, file, indent=4)

def run_best_model(model, X, y, model_name):
    pass

def main(args):
    X, y, test_fold = load_data(args.filepath, feature=args.feature)
    evaluate_models(X, y, test_fold, model_names=args.models)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models on the QM7 dataset.")
    parser.add_argument("--filepath", 
                        type=str,
                        default="../../data/raw/qm7.mat",)
    parser.add_argument("--feature", 
                        type=str, 
                        default="eigenspectrum", 
                        )
    parser.add_argument("--models", 
                        type=str, 
                        nargs="+", default=["Linear"]
                        )
    
    args = parser.parse_args()
    main(args)
