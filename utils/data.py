

def get_cv_fold(dataset: dict, fold):
    
    ids_train = dataset['P'][list(range(0, fold)) + list(range(fold+1, 5))].flatten()
    ids_val = dataset['P'][list(range(fold, fold+1))].flatten()

    dataset["T"] = dataset["T"].squeeze()
    data_train = {
        "X": dataset["X"][ids_train],
        "T": dataset["T"][ids_train],
        "Z": dataset["Z"][ids_train],
        "R": dataset["R"][ids_train],
    }
    data_val = {
        "X": dataset["X"][ids_val],
        "T": dataset["T"][ids_val],
        "Z": dataset["Z"][ids_val],
        "R": dataset["R"][ids_val],
    }

    return data_train, data_val