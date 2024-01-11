from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import pandas as pd
from typing import Dict, Any
import numpy as np


def tune_random_forest_hyperparameters():
    """
    ランダムフォレストモデルを訓練し、評価する。

    Args:
        X_train, y_train: 学習用データとラベル。
        X_test, y_test: 評価用データとラベル。

    Returns:
        model: 訓練されたランダムフォレストモデル。
    """
    # 特徴行列の保存先のpath
    feature_matrix_path = "feature\\feature_matrix\\feature_matrix.ftr"
    labels_path = "feature\\feature_matrix\\labels.ftr"

    # データをランダムに分割
    X_train, X_test, y_train, y_test = _load_data(feature_matrix_path, labels_path, test_size=0.2)
    y_train, y_test  = np.ravel(y_train), np.ravel(y_test)

    best_params = _set_random_forest_hyperparameters(X_train, y_train)
    print("Best Parameters:", best_params)

def _set_random_forest_hyperparameters(X_train: pd.DataFrame,
                                      y_train: pd.Series,
                                      n_iter: int = 2000,
                                      cv: int = 2) -> Dict[str, Any]:
    """
    ランダムフォレストのハイパーパラメータチューニングを行う。

    Args:
        X_train (pd.DataFrame): トレーニングデータの特徴量。
        y_train (pd.Series): トレーニングデータのターゲット変数。
        n_iter (int, optional): ランダムサーチの試行回数。デフォルトは100。
        cv (int, optional): 交差検証の分割数。デフォルトは5。

    Returns:
        Dict[str, Any]: 最適なハイパーパラメータのセット。
    """
    # ランダムフォレスト分類器のインスタンス化
    rf = RandomForestClassifier(random_state=0)

    # ハイパーパラメータの分布を定義
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_features': ['sqrt', 'log2'],
        'max_depth': randint(4, 100),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 11),
        'bootstrap': [True, False],
        "criterion": ["gini", "entropy"]}

    # ランダムサーチの実行
    random_search = RandomizedSearchCV(rf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter,
                                       scoring="accuracy",
                                       cv=cv,
                                       verbose=3,
                                       n_jobs=-1)

    random_search.fit(X_train, y_train)

    # 最適なパラメータの返却
    return random_search.best_params_

def _load_data(feature_matrix_path, labels_path, test_size: float = 0.05):
    """
    特徴行列を読み込み、学習用データと評価用データに分割する。

    Args:
        file_path (str): 特徴行列のファイルパス。
        test_size (float): テストデータの割合。

    Returns:
        X_train, X_test, y_train, y_test: 分割された学習用データと評価用データ。
    """
    feature_matrix = pd.read_feather(feature_matrix_path).T
    feature_matrix = feature_matrix.to_numpy()
    labels = pd.read_feather(labels_path).T
    labels = labels.to_numpy()

    return train_test_split(feature_matrix, labels, test_size=test_size, random_state=0, stratify=labels)
