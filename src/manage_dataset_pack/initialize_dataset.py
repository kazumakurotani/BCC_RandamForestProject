import shutil
import os

def initialize_dataset():
    # data//raw//imagesを除く画像データを全削除
    dataset_paths =[
        "data\\raw\\labeled",
        "data\\generated",
        "data\\preprocessed",
    ]

    feature_paths = [
        "feature\\feature_matrix",
        "feature\\overall"
    ]

    for dataset_path in dataset_paths:
        if os.path.isdir(dataset_path) is True:
            for dir_path in os.listdir(dataset_path):
                path = os.path.join(dataset_path, dir_path)
                shutil.rmtree(path)
        else:
            continue

    for feature_path in feature_paths:
        if os.path.isdir(feature_path) is True:
            shutil.rmtree(feature_path)
        else:
            continue
