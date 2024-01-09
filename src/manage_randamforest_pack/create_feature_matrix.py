import pandas as pd
from tqdm import tqdm
import shutil

def create_feature_matrix() -> None:
    """
    アクティブな特徴量ファイルを読み込み、連結してfeather形式で保存する。
    """
    active_features = {
        "grayscale": "feature\\overall\\grayscale_feature_overall.ftr"
    }
    labels_path = "feature\\overall\\labels_overall.ftr"

    # feature_matrixの読み込み
    feature_matrix = _load_features(active_features)

    # feature_matrix をfeather形式で保存
    feature_matrix.to_feather('feature\\feature_matrix\\feature_matrix.ftr')
    shutil.copy(labels_path, "feature\\feature_matrix")

def _load_features(features: dict) -> pd.DataFrame:
    """
    指定された特徴量ファイルを読み込み、連結する。

    Args:
        features (dict): 特徴量名とファイルパスをマッピングする辞書。

    Returns:
        pd.DataFrame: 連結された特徴量データ。
    """
    concatenated_feature_matrix = None
    feature_ranges = []
    current_start = 0

    for feature_name, file_path in tqdm(features.items(), desc="Creating Feature Matrix"):
        # 特徴量を読み込む
        feature_df = pd.read_feather(file_path)

        # 特徴量の範囲を記録する
        current_end = current_start + feature_df.shape[0] - 1
        feature_ranges.append([current_start, current_end, feature_name])
        current_start = current_end + 1

        # 特徴量を連結する
        if concatenated_feature_matrix is None:
            concatenated_feature_matrix = feature_df
        else:
            concatenated_feature_matrix = pd.concat([concatenated_feature_matrix, feature_df], axis=0, ignore_index=True)

    # 特徴量の範囲をCSVファイルに保存
    feature_ranges_df = pd.DataFrame(feature_ranges, columns=['Start', 'End', 'FeatureName'])
    feature_ranges_df.to_csv('feature\\feature_matrix\\feature_ranges.csv', index=False)

    return concatenated_feature_matrix

