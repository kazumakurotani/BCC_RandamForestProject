from src import manage_dataset

class Main():
    def __init__(self) -> None:
        # 操作内容を選択
        self.options = {
            0: "labeling_image",
            1: "generating",
            999: "initializing"
        }

        # 処理内容の選択
        self.is_select_option = 999

        # 処理項目の選択
        self.is_select_class_index = "class2"

        # generationg_imageのオプション
        self.augmentation_option = 0 # 0: shift and rotate 1:rotate

        # インスタンスの生成
        self.manager = manage_dataset.DataSetManager()

    def labeling_image(self):
        print("Start Labeling Image")
        self.manager.manage_dataset("labeling", self.is_select_class_index)

    def generating_image(self):
        print("Start Generating Image")
        self.manager.manage_dataset("generating", self.augmentation_option)

    def initializing_dataset(self):
        print("Initializing Dataset")
        self.manager.manage_dataset("initializing")

    def message(self):
        print("処理が完了しました")

    def main(self):
        if self.is_select_option == 0:
            self.labeling_image()
        elif self.is_select_option == 1:
            self.generating_image()
        elif self.is_select_option == 999:
            self.initializing_dataset()
        self.message()

if __name__ == "__main__":
    Main = Main()
    Main.main()
