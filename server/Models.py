from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
from common import getInterval
from scipy.stats import t



def KNN(x_train_scaled, y_train, n_neighbors, weights, p):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    precisions_per_class, recalls_per_class, f1_scores_per_class = [], [], []
    conf_matrices, accuracies = [], []

    for train_idx, val_idx in kf.split(x_train_scaled, y_train):
        X_train_fold = x_train_scaled[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = x_train_scaled[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p, n_jobs=-1)
        knn.fit(X_train_fold, y_train_fold)
        y_pred = knn.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        rec = recall_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        f1 = f1_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        cm = confusion_matrix(y_val_fold, y_pred)

        accuracies.append(acc)
        precisions_per_class.append(prec)
        recalls_per_class.append(rec)
        f1_scores_per_class.append(f1)
        conf_matrices.append(cm)
    
    precisions_per_class = np.array(precisions_per_class)
    recalls_per_class = np.array(recalls_per_class)
    f1_scores_per_class = np.array(f1_scores_per_class)
    conf_matrix_avg = np.mean(conf_matrices, axis=0).astype(int)

    conf_level = 0.95
    dof = len(accuracies) - 1
    t_value = t.ppf((1 + conf_level) / 2, dof)

    val = t_value * np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))

    interval = getInterval(accuracies, precisions_per_class, recalls_per_class, f1_scores_per_class, val)

    return {
        'precision': precisions_per_class.mean(axis=0).tolist(),
        'recall': recalls_per_class.mean(axis=0).tolist(),
        'f1_score': f1_scores_per_class.mean(axis=0).tolist(),
        'accuracy': np.mean(accuracies),
        'conf_matrix': conf_matrix_avg.tolist(),
        'interval': interval
    }


def MLP(x_train_scaled, y_train, hidden_layer_sizes, activation, solver, max_iter):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    precisions_per_class, recalls_per_class, f1_scores_per_class = [], [], []
    conf_matrices, accuracies = [], []

    for train_idx, val_idx in kf.split(x_train_scaled, y_train):
        X_train_fold = x_train_scaled[train_idx]
        y_train_fold = y_train.iloc[train_idx]

        X_val_fold = x_train_scaled[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter, random_state=42)
        mlp.fit(X_train_fold, y_train_fold)

        y_pred = mlp.predict(X_val_fold)

        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        rec = recall_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        f1 = f1_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        cm = confusion_matrix(y_val_fold, y_pred)

        accuracies.append(acc)
        precisions_per_class.append(prec)
        recalls_per_class.append(rec)
        f1_scores_per_class.append(f1)
        conf_matrices.append(cm)

    
    precisions_per_class = np.array(precisions_per_class)
    recalls_per_class = np.array(recalls_per_class)
    f1_scores_per_class = np.array(f1_scores_per_class)
    conf_matrix_avg = np.mean(conf_matrices, axis=0).astype(int)

    conf_level = 0.95
    dof = len(accuracies) - 1
    t_value = t.ppf((1 + conf_level) / 2, dof)
    val = t_value * np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))

    interval = getInterval(accuracies, precisions_per_class, recalls_per_class, f1_scores_per_class, val)

    return {
        'precision': precisions_per_class.mean(axis=0).tolist(),
        'recall': recalls_per_class.mean(axis=0).tolist(),
        'f1_score': f1_scores_per_class.mean(axis=0).tolist(),
        'accuracy': np.mean(accuracies),
        'conf_matrix': conf_matrix_avg.tolist(),
        'interval': interval
    }


def XGB(x_train_scaled, y_train, n_estimators, scale_pos_weight, eval_metric, n_jobs):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    precisions_per_class, recalls_per_class, f1_scores_per_class = [], [], []
    conf_matrices, accuracies = [], []

    for train_idx, val_idx in kf.split(x_train_scaled, y_train):
        X_train_fold = x_train_scaled[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = x_train_scaled[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        xgb_model_cv = XGBClassifier(
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight,
            eval_metric=eval_metric,
            n_jobs=n_jobs,
            random_state=42
        )
        xgb_model_cv.fit(X_train_fold, y_train_fold)
        y_pred = xgb_model_cv.predict(X_val_fold)

        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        rec = recall_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        f1 = f1_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        cm = confusion_matrix(y_val_fold, y_pred)

        accuracies.append(acc)
        precisions_per_class.append(prec)
        recalls_per_class.append(rec)
        f1_scores_per_class.append(f1)
        conf_matrices.append(cm)

    precisions_per_class = np.array(precisions_per_class)
    recalls_per_class = np.array(recalls_per_class)
    f1_scores_per_class = np.array(f1_scores_per_class)
    conf_matrix_avg = np.mean(conf_matrices, axis=0).astype(int)

    conf_level = 0.95
    dof = len(accuracies) - 1
    t_value = t.ppf((1 + conf_level) / 2, dof)

    val = t_value * np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))

    interval = getInterval(accuracies, precisions_per_class, recalls_per_class, f1_scores_per_class, val)

    return {
        'precision': precisions_per_class.mean(axis=0).tolist(),
        'recall': recalls_per_class.mean(axis=0).tolist(),
        'f1_score': f1_scores_per_class.mean(axis=0).tolist(),
        'accuracy': np.mean(accuracies),
        'conf_matrix': conf_matrix_avg.tolist(),
        'interval': interval
    }


def DecisionTree(x_train_scaled, y_train):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    precisions_per_class, recalls_per_class, f1_scores_per_class = [], [], []
    conf_matrices, accuracies = [], []

    for train_idx, val_idx in kf.split(x_train_scaled, y_train):
        X_train_fold = x_train_scaled[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = x_train_scaled[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train_fold, y_train_fold)
        y_pred = dt_model.predict(X_val_fold)

        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        rec = recall_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        f1 = f1_score(y_val_fold, y_pred, average=None, labels=[0, 1])
        cm = confusion_matrix(y_val_fold, y_pred)

        accuracies.append(acc)
        precisions_per_class.append(prec)
        recalls_per_class.append(rec)
        f1_scores_per_class.append(f1)
        conf_matrices.append(cm)

    precisions_per_class = np.array(precisions_per_class)
    recalls_per_class = np.array(recalls_per_class)
    f1_scores_per_class = np.array(f1_scores_per_class)
    conf_matrix_avg = np.mean(conf_matrices, axis=0).astype(int)

    conf_level = 0.95
    dof = len(accuracies) - 1
    t_value = t.ppf((1 + conf_level) / 2, dof)
    val = t_value * np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))

    interval = getInterval(accuracies, precisions_per_class, recalls_per_class, f1_scores_per_class, val)

    return {
        'precision': precisions_per_class.mean(axis=0).tolist(),
        'recall': recalls_per_class.mean(axis=0).tolist(),
        'f1_score': f1_scores_per_class.mean(axis=0).tolist(),
        'accuracy': np.mean(accuracies),
        'conf_matrix': conf_matrix_avg.tolist(),
        'interval': interval
    }


def trainModel(model_name, x_train_scaled, y_train, params):
    if model_name == 'KNN':
        return KNN(
            x_train_scaled,
            y_train,
            n_neighbors=params.get('n_neighbors', 5),
            weights=params.get('weights', 'uniform'),
            p=params.get('p', 2)
        )
    elif model_name == 'MLP':
        return MLP(
            x_train_scaled,
            y_train,
            hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
            activation=params.get('activation', 'relu'),
            solver=params.get('solver', 'adam'),
            max_iter=params.get('max_iter', 200)
        )
    elif model_name == 'XGB':
        return XGB(
            x_train_scaled,
            y_train,
            n_estimators=params.get('n_estimators', 100),
            scale_pos_weight=params.get('scale_pos_weight', 1),
            eval_metric=params.get('eval_metric', 'logloss'),
            n_jobs=params.get('n_jobs', -1)
        )
    elif model_name == 'DecisionTree':
        return DecisionTree(
            x_train_scaled,
            y_train
        )
    else:
        raise ValueError(f"Model {model_name} is not supported.")