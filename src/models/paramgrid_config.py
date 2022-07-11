import numpy as np

def grid_random_forest():
    # Number of trees in random forest
    n_estimators = list(range(50, 500, 10))
    # Number of features to consider at every split
    max_features = list(np.arange(1, 15,1))
    # Maximum number of levels in tree
    max_depth = list(range(5, 100, 1))
    # Minimum number of samples required to split a node
    min_samples_split = list(np.arange(1, 15,1))
    # Minimum number of samples required at each leaf node
    min_samples_leaf = list(np.arange(1, 15,1))

    param_grid = {'model__n_estimators': n_estimators,
                'model__max_features': max_features,
                'model__max_depth': max_depth,
                'model__min_samples_split': min_samples_split,
                'model__min_samples_leaf': min_samples_leaf}

    return param_grid

def grid_xgboost():
    param_grid = {
    "model__n_estimators":list(range(50, 500, 1)),
    'model__max_depth': list(range(2, 50)),
    "model__learning_rate": list(np.arange(0.01, 0.55,0.01)),
    "model__gamma": list(np.arange(0, 1.5, 0.1)),
    "model__reg_alpha": list(np.arange(0, 1.5, 0.05)),
    "model__reg_lambda": list(np.arange(0, 7, 0.1)),
    "model__scale_pos_weight": list(np.arange(0.1, 1.5, 0.1)),
    "model__subsample": list(np.arange(0.1, 2.0, 0.1)),
    "model__min_child_weight":list(np.arange(1, 7,1))}

    return param_grid

