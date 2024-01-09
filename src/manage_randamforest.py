from src.manage_randamforest_pack import create_feature_matrix, train_and_evaluate, plot_results


class RandamForestManager:
    def __init__(self):
        pass

    def manage_randamforest(self, operation, *args, **kwargs):
        # 操作に基づいて子モジュールの関数を呼び出す
        if operation == "create_feature_matrix":
            return create_feature_matrix.create_feature_matrix(*args, **kwargs)
        elif operation == "train_and_evaluate":
            return train_and_evaluate.train_and_evaluate(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
