import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def train_and_evaluate():
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
    x_train, x_test, y_train, y_test = _load_data(feature_matrix_path, labels_path, test_size=0.2)
    y_train, y_test  = np.ravel(y_train), np.ravel(y_test)

    model = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth= 85, max_features= "sqrt", min_samples_leaf= 1, min_samples_split= 3, n_estimators= 500, random_state=0, verbose=2)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    return model


def _load_data(feature_matrix_path, labels_path, test_size: float = 0.2):
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

    return train_test_split(feature_matrix, labels, test_size=test_size, random_state=42, stratify=labels)
