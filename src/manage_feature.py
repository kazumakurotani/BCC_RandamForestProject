from src.manage_feature_pack import feature_engineering, preprocess


class FeatureManager:
    def __init__(self):
        pass

    def manage_feature(self, operation, *args, **kwargs):
        # 操作に基づいて子モジュールの関数を呼び出す
        if operation == "feature_engineering":
            return feature_engineering.generate_features(*args, **kwargs)
        elif operation == "preprocessing_image":
            return preprocess.preprocess(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
