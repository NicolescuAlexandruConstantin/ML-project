import pandas as pd
import numpy as np

from knn import KNNRegressorScratch, StandardScalerScratch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

DATA_PATH = "earthquake_1995-2023.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

alert_mapping = {
    'none': 0,
    'green': 1,
    'yellow': 2,
    'orange': 3,
    'red': 4
}

df['alert_encoded'] = df['alert'].fillna('none').map(alert_mapping)

df_clean = df.select_dtypes(include=[np.number])

X = df_clean.drop('sig', axis=1).values
y = df_clean['sig'].values

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def normalized_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred) / (np.max(y_true) - np.min(y_true))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def k_fold_split(X, y, k_folds=10, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    folds = np.array_split(indices, k_folds)

    for i in range(k_folds):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k_folds) if j != i])

        yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def evaluate_knn_cv(X, y, k, k_folds=10):
    mae_scores, rmse_scores, nrmse_scores, r2_scores = [], [], [], []

    for X_train, X_test, y_train, y_test in k_fold_split(X, y, k_folds):
        scaler = StandardScalerScratch()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = KNNRegressorScratch(k=k)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(root_mean_squared_error(y_test, y_pred))
        nrmse_scores.append(normalized_rmse(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    return {
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores),
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
        "NRMSE_mean": np.mean(nrmse_scores),
        "R2_mean": np.mean(r2_scores)
    }


def grid_search_knn(X, y, k_values, k_folds=10):
    return {k: evaluate_knn_cv(X, y, k, k_folds) for k in k_values}

def evaluate_sklearn_knn_cv(X, y, k, k_folds=10):
    mae_scores, rmse_scores, nrmse_scores, r2_scores = [], [], [], []

    for X_train, X_test, y_train, y_test in k_fold_split(X, y, k_folds):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = KNeighborsRegressor(n_neighbors=k, weights="distance")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(root_mean_squared_error(y_test, y_pred))
        nrmse_scores.append(normalized_rmse(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    return {
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores),
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
        "NRMSE_mean": np.mean(nrmse_scores),
        "R2_mean": np.mean(r2_scores)
    }


def grid_search_sklearn_knn(X, y, k_values, k_folds=10):
    return {k: evaluate_sklearn_knn_cv(X, y, k, k_folds) for k in k_values}

def confidence_interval(mean, std, k_folds=10):
    margin = 1.96 * std / np.sqrt(k_folds)
    return mean - margin, mean + margin

k_values = [3, 5, 7, 11, 15]

scratch_results = grid_search_knn(X, y, k_values)
sklearn_results = grid_search_sklearn_knn(X, y, k_values)

print("\n===== SCRATCH KNN RESULTS =====")
for k, metrics in scratch_results.items():
    print(f"\nk = {k}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

print("\n===== SKLEARN KNN RESULTS =====")
for k, metrics in sklearn_results.items():
    print(f"\n[sklearn] k = {k}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

best_k_scratch = min(scratch_results, key=lambda k: scratch_results[k]["RMSE_mean"])
best_k_sklearn = min(sklearn_results, key=lambda k: sklearn_results[k]["RMSE_mean"])

print("\nBest Scratch k =", best_k_scratch)
print("Best sklearn k =", best_k_sklearn)

rmse_mean_scratch = scratch_results[best_k_scratch]["RMSE_mean"]
rmse_std_scratch = scratch_results[best_k_scratch]["RMSE_std"]

rmse_mean_skl = sklearn_results[best_k_sklearn]["RMSE_mean"]
rmse_std_skl = sklearn_results[best_k_sklearn]["RMSE_std"]

ci_scratch = confidence_interval(rmse_mean_scratch, rmse_std_scratch)
ci_sklearn = confidence_interval(rmse_mean_skl, rmse_std_skl)

print(f"\n95% CI Scratch RMSE: [{ci_scratch[0]:.2f}, {ci_scratch[1]:.2f}]")
print(f"95% CI sklearn RMSE: [{ci_sklearn[0]:.2f}, {ci_sklearn[1]:.2f}]")
