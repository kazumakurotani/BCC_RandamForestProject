import os
import cv2
from src.manage_feature_pack import feather
import pandas as pd
from tqdm import tqdm


def generate_features():
    # 画像データと行列の対応表の作成

    # 特徴生成の実行
    args = feather.get_arguments("test")
    feather.generate_features(globals(), args.force)


class Labels(feather.Feather):
    def create_features(self):
        """
        画像のラベルをエンコードし、特徴量をDataFrameに保存する。

        各画像に対して一意の整数ラベルを割り当て、それらの特徴量と名前の対応関係を保存する。
        """
        features = {}
        names_correspondence = {}

        # ラベル名の重複を除去
        seen = set()
        unique_items = [label_name for label_name in self.label_names if not (label_name in seen or seen.add(label_name))]

        # ラベルエンコーディング
        encoded_labels = {item: i for i, item in enumerate(unique_items)}

        # パスから画像名を抽出
        image_names = [os.path.basename(path) for path in self.image_paths]

        try:
            for i, path in tqdm(enumerate(self.image_paths), desc="Labels"):
                dir_path = os.path.dirname(path)
                dir_name = os.path.basename(dir_path)
                features[f"col_{i}"] = [encoded_labels[dir_name]]
                names_correspondence[f"col_{i}"] = image_names[i]

            new_df = pd.DataFrame(features)
            self.overall = pd.concat([self.overall, new_df], ignore_index=True)
            self.create_memo("The correct label of the image.")
            self.create_names_correspondence(names_correspondence)
        except Exception as e:
            print(f"An error occurred: {e}")


class GrayscaleFeature(feather.Feather):
    def create_features(self):
        """
        画像をグレースケールに変換し、平坦化して特徴量を抽出する。

        各画像をグレースケールに変換した後、一次元配列に平坦化し、Pandas DataFrameに格納する。
        """
        features = {}
        names_correspondence = {}

        for i, path in tqdm(enumerate(self.image_paths), desc="GrayscaleFeature"):
            image_name = os.path.basename(path)

            try:
                image = self.load_image(path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                features[f"col_{i}"] = gray_image.flatten()
                names_correspondence[f"col_{i}"] = image_name
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue

        new_df = pd.DataFrame(features)
        self.overall = pd.concat([self.overall, new_df], ignore_index=True)
        self.create_memo("Convert an image to grayscale and then flatten it into a one-dimensional array.")
        self.create_names_correspondence(names_correspondence)


"""def save_index_image_map_to_csv() -> None:

    Save the index-image name mapping to a CSV file.

    csv_file_path = os.path.join("feature", '_features.csv')

    for i, path in tqdm(enumerate(self.image_paths), desc="GrayscaleFeature"):
        index_image_name_map = {i: os.path.basename(path) for i, path in enumerate(image_paths)}

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Image Name'])  # Writing header
        for index, image_name in index_image_name_map.items():
            writer.writerow([index, image_name])"""
