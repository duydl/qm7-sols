import os
import json
import argparse
import numpy as np
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from helper import load_data, numpy_to_python
from models import get_models

def run_best_model(model, X, y, model_name):
    pass

def main(args):
    X, y, test_fold = load_data(args.data_path, feature=args.feature)
    
    models = get_models(model_names=args.models)
    results_file_path = f"{args.log_dir}/{args.feature}_grid_search_results.json"
    log_file_path = f"{args.log_dir}/{args.feature}_grid_search_log.txt"
    
    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as file:
            results = json.load(file)
    else:
        results = {}
        
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
            verbose=2,
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
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Best model for {model_name}: {best_model}, Best MAE: {best_score}\n")
    
    with open(results_file_path, "w") as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models on the QM7 dataset.")
    parser.add_argument("--feature", 
                        type=str, 
                        default="eigenspectrum", 
                        )
    parser.add_argument("--models", 
                        type=str, 
                        nargs="+", default=["Linear"]
                        )
    parser.add_argument('--log_dir', 
                        type=str, 
                        default='logs'
                        )
    parser.add_argument('--mlflow_uri', 
                        type=str, 
                        default=os.path.expanduser('~/mlruns'),)
    parser.add_argument('--data_path', 
                        type=str, 
                        default='../../data/raw/qm7.mat')
    
    args = parser.parse_args()
    main(args)
