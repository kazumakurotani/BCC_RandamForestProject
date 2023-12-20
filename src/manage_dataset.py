from src.manage_dataset_pack import labeling_image


class DataSetManager:
    def __init__(self):
        pass

    def manage_dataset(self, operation, *args, **kwargs):
        # 操作に基づいて子モジュールの関数を呼び出す
        if operation == "labeling":
            return labeling_image.labeling_image(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
