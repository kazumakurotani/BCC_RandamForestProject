from src import manage_dataset, manage_feature, manage_randamforest


class Main():
    def __init__(self) -> None:
        # 操作内容を選択
        self.options = {
            0: "labeling_image",
            1: "generating",
            2: "preprocessing",
            3: "feature_engineering",
            4: "create_feature_matrix",
            5: "train_and_evaluate",
            999: "initializing"
        }

        # 処理内容の選択
        self.is_select_option = 999

        #Dataset#########################################################################

        # 処理項目の選択
        self.is_select_class_index = "class3"

        # generationg_imageのオプション
        self.augmentation_option = 1 # 0: shift and rotate 1:rotate

        ################################################################################

        #Feature#########################################################################

        #


        ################################################################################

        # インスタンスの生成
        self.dm = manage_dataset.DataSetManager()
        self.fm = manage_feature.FeatureManager()
        self.rfm = manage_randamforest.RandamForestManager()


    def labeling_image(self):
        print("Start Labeling Image")
        self.dm.manage_dataset("labeling", self.is_select_class_index)

    def generating_image(self):
        print("Start Generating Image")
        self.dm.manage_dataset("generating", self.augmentation_option)

    def proprocessing_image(self):
        print("Start Proprocessing Image")
        self.fm.manage_feature("preprocessing_image")

    def feature_engineering(self):
        print("Start Feature Engineering")
        self.fm.manage_feature("feature_engineering")

    def create_feature_matrix(self):
        print("Start Creating Feature Matrix")
        self.rfm.manage_randamforest("create_feature_matrix")

    def train_and_evaluate(self):
        print("Start Training and Evaluating")
        self.rfm.manage_randamforest("train_and_evaluate")

    def initializing_dataset(self):
        print("Initializing Dataset")
        self.dm.manage_dataset("initializing")

    def message(self):
        print("処理が完了しました")

    def main(self):
        if self.is_select_option == 0:
            self.labeling_image()
        elif self.is_select_option == 1:
            self.generating_image()
        elif self.is_select_option == 2:
            self.proprocessing_image()
        elif self.is_select_option == 3:
            self.feature_engineering()
        elif self.is_select_option == 4:
            self.create_feature_matrix()
        elif self.is_select_option == 5:
            self.train_and_evaluate()
        elif self.is_select_option == 999:
            self.initializing_dataset()
        self.message()

if __name__ == "__main__":
    Main = Main()
    Main.main()
