import shutil

def initialize_dataset():
    # data//raw//imagesを除く画像データを全削除
    dataset_paths =[
        "data\\raw\\labeled",
        "data\\generated",
        "feature\\overall",
        "feature\\feature_matrix"
    ]

    for dataset_path in dataset_paths:
        shutil.rmtree(dataset_path)
