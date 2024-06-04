import numpy as np
import scipy

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
    upper_flatten_cm = np.array([
        cm[np.triu_indices_from(cm)] for cm in sorted_coulomb_matrices
    ])
    return upper_flatten_cm

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
