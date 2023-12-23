import shutil

def initialize_dataset():
    # data//raw//imagesを簿族データを全削除
    dataset_paths =[
        "data\\raw\\labeled",
        "data\\generated"
    ]

    for dataset_path in dataset_paths:
        shutil.rmtree(dataset_path)
