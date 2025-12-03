import pandas as pd
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import numpy as np

def GIWRF(df_train, df_test):
    X = df_train.drop(columns=['label'])
    y = df_train['label']

    giwrf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    giwrf.fit(X, y)

    feature_importances = pd.Series(giwrf.feature_importances_, index=X.columns)
    threshold = 0.02
    selected_features = feature_importances[feature_importances > threshold].index.tolist()

    df_train = df_train[selected_features + ['label']]
    df_test = df_test[selected_features + ['label']]

    return df_train, df_test


def scale_data(df_train, df_test):
    scaler = MinMaxScaler()
    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns=['label'])
    y_test = df_test['label']

    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    return x_train_scaled, y_train, x_test_scaled, y_test


def preprocess(df):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    malign_train = df_train[df_train['label'] == 1]
    benign_train = df_train[df_train['label'] == 0]
    n = min(len(malign_train), len(benign_train))
    malign_train = resample(malign_train, replace=True, n_samples=n, random_state=42)
    benign_train = resample(benign_train, replace=True, n_samples=n, random_state=42)
    df_train = pd.concat([malign_train, benign_train]).sample(frac=1, random_state=42).reset_index(drop=True)

    malign_test = df_test[df_test['label'] == 1]
    benign_test = df_test[df_test['label'] == 0]
    n = min(len(malign_test), len(benign_test))
    malign_test = resample(malign_test, replace=True, n_samples=n, random_state=42)
    benign_test = resample(benign_test, replace=True, n_samples=n, random_state=42)
    df_test = pd.concat([malign_test, benign_test]).sample(frac=1, random_state=42).reset_index(drop=True)

    transform_cols = ['proto', 'state', 'service']
    encoder = TargetEncoder(cols=transform_cols, handle_unknown='value')

    df_train_encoded = encoder.fit_transform(df_train[transform_cols], df_train['label'])
    df_train_rest = df_train.drop(columns=transform_cols)
    df_train = pd.concat([df_train_rest, df_train_encoded], axis=1)

    df_test_encoded = encoder.transform(df_test[transform_cols])
    df_test_rest = df_test.drop(columns=transform_cols)
    df_test = pd.concat([df_test_rest, df_test_encoded], axis=1)

    corr_matrix_abs = df_train.corr().abs()
    upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
    if 'label' in to_drop:
        to_drop.remove('label')
    df_train = df_train.drop(columns=to_drop)
    df_test = df_test.drop(columns=to_drop)

    df_train, df_test = GIWRF(df_train, df_test)

    X_train_scaled, y_train, X_test_scaled, y_test = scale_data(df_train, df_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


def getInterval(accuracies, precisions_per_class, recalls_per_class, f1_scores_per_class, val):
    interval_acc = [
    max(0,round(np.mean(accuracies) - val, 4)),
    min(1,round(np.mean(accuracies) + val, 4))
    ]

    interval_f1_normal = [
    max(0, round(np.mean(f1_scores_per_class[:, 0]) - val, 4)),
    min(1, round(np.mean(f1_scores_per_class[:, 0]) + val, 4))
    ]

    interval_f1_attack = [
    max(0, round(np.mean(f1_scores_per_class[:, 1]) - val, 4)),
    min(1, round(np.mean(f1_scores_per_class[:, 1]) + val, 4))
    ]
    
    interval_precision_normal = [
    max(0, round(np.mean(precisions_per_class[:, 0]) - val, 4)),
    min(1, round(np.mean(precisions_per_class[:, 0]) + val, 4))
    ]
    
    interval_precision_attack = [
    max(0, round(np.mean(precisions_per_class[:, 1]) - val, 4)),
    min(1, round(np.mean(precisions_per_class[:, 1]) + val, 4))
    ]
    
    interval_recall_normal = [
    max(0, round(np.mean(recalls_per_class[:, 0]) - val, 4)),
    min(1, round(np.mean(recalls_per_class[:, 0]) + val, 4))
    ]
    
    interval_recall_attack = [
    max(0, round(np.mean(recalls_per_class[:, 1]) - val, 4)),
    min(1, round(np.mean(recalls_per_class[:, 1]) + val, 4))
    ]

    return{
        'accuracy': interval_acc,
        'f1_score_normal': interval_f1_normal,
        'f1_score_attack': interval_f1_attack,
        'precision_normal': interval_precision_normal,
        'precision_attack': interval_precision_attack,
        'recall_normal': interval_recall_normal,
        'recall_attack': interval_recall_attack
    }