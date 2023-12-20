from src import manage_dataset

class Main():
    def __init__(self) -> None:
        # 操作内容を選択
        self.options = {
            0: "labeling_image"
        }

        # 処理内容の選択
        self.is_select_option = 0

        # 処理項目の選択
        self.is_select_class_index = "class2"

    def labeling_image(self):
        manager = manage_dataset.DataSetManager()

        manager.manage_dataset("labeling", self.is_select_class_index)

    def message(self):
        print("処理が完了しました")

    def main(self):
        if self.is_select_option == 0:
            self.labeling_image()
            self.message()

if __name__ == "__main__":
    Main = Main()
    Main.main()
