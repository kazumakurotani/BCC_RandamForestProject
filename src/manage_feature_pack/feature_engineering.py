import os
import cv2
from src.manage_feature_pack import feather
import pandas as pd
from tqdm import tqdm
import numpy as np

from skimage import feature


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
            for i in tqdm(range(len(self.image_paths)), desc="Labels"):
                path = self.image_paths[i]
                dir_path = os.path.dirname(path)
                dir_name = os.path.basename(dir_path)
                features[str(i).zfill(6)] = [encoded_labels[dir_name]]
                names_correspondence[str(i).zfill(6)] = image_names[i]

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

        for i in tqdm(range(len(self.image_paths)), desc="GrayscaleFeature"):
            path = self.image_paths[i]
            image_name = os.path.basename(path)

            try:
                image = self.load_image(path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                features[str(i).zfill(6)] = gray_image.flatten()
                names_correspondence[str(i).zfill(6)] = image_name
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue

        new_df = pd.DataFrame(features)
        self.overall = pd.concat([self.overall, new_df], ignore_index=True)
        self.create_memo("Convert an image to grayscale and then flatten it into a one-dimensional array.")
        self.create_names_correspondence(names_correspondence)


class LbpGray(feather.Feather):
    """
    画像を読み込み、グレースケールに変換後、Local Binary Pattern 特徴量を計算する。

    Args:
        image_path (str): 画像のファイルパス。

    Returns:
        np.ndarray: LBP特徴量の一次元配列。

    Raises:
        IOError: 画像ファイルの読み込みに失敗した場合に発生。
    """
    def create_features(self):
        features = {}
        names_correspondence = {}

        for i in tqdm(range(len(self.image_paths)), desc="LBP for Gray"):
            path = self.image_paths[i]
            image_name = os.path.basename(path)

            try:
                # 画像を読み込み、グレースケールに変換
                image = self.load_image(path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # LBP特徴量を計算
                points, radius = 16, 2
                lbp = feature.local_binary_pattern(gray_image, points, radius, method="uniform")

                # ヒストグラムを計算
                hist, _ = np.histogram(lbp.ravel(), bins=points+2, range=(0, points+2), density=True)

                features[str(i).zfill(6)] = hist
                names_correspondence[str(i).zfill(6)] = image_name
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue

        new_df = pd.DataFrame(features)
        self.overall = pd.concat([self.overall, new_df], ignore_index=True)
        self.create_memo("Convert an image to grayscale, compute Local Binary Patterns, and then flatten them into a one-dimensional array.")
        self.create_names_correspondence(names_correspondence)


class LbpRed(feather.Feather):
    """
    画像を読み込み、R成分に変換後、Local Binary Pattern 特徴量を計算する。

    Args:
        image_path (str): 画像のファイルパス。

    Returns:
        np.ndarray: LBP特徴量の一次元配列。

    Raises:
        IOError: 画像ファイルの読み込みに失敗した場合に発生。
    """
    def create_features(self):
        features = {}
        names_correspondence = {}

        for i in tqdm(range(len(self.image_paths)), desc="LBP for Red"):
            path = self.image_paths[i]
            image_name = os.path.basename(path)

            try:
                # 画像を読み込み、グレースケールに変換
                image = self.load_image(path)
                red_image = image[:, :, 2]

                # LBP特徴量を計算
                points, radius = 16, 2
                lbp = feature.local_binary_pattern(red_image, points, radius, method="uniform")

                # ヒストグラムを計算
                hist, _ = np.histogram(lbp.ravel(), bins=points+2, range=(0, points+2), density=True)

                features[str(i).zfill(6)] = hist
                names_correspondence[str(i).zfill(6)] = image_name
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue

        new_df = pd.DataFrame(features)
        self.overall = pd.concat([self.overall, new_df], ignore_index=True)
        self.create_memo("Convert an image to R component, compute Local Binary Patterns, and then flatten them into a one-dimensional array.")
        self.create_names_correspondence(names_correspondence)


class LbpGreen(feather.Feather):
    """
    画像を読み込み、G成分に変換後、Local Binary Pattern 特徴量を計算する。

    Args:
        image_path (str): 画像のファイルパス。

    Returns:
        np.ndarray: LBP特徴量の一次元配列。

    Raises:
        IOError: 画像ファイルの読み込みに失敗した場合に発生。
    """
    def create_features(self):
        features = {}
        names_correspondence = {}

        for i in tqdm(range(len(self.image_paths)), desc="LBP for Green"):
            path = self.image_paths[i]
            image_name = os.path.basename(path)

            try:
                # 画像を読み込み、グレースケールに変換
                image = self.load_image(path)
                converted_image = image[:, :, 1]

                # LBP特徴量を計算
                points, radius = 16, 2
                lbp = feature.local_binary_pattern(converted_image, points, radius, method="uniform")

                # ヒストグラムを計算
                hist, _ = np.histogram(lbp.ravel(), bins=points+2, range=(0, points+2), density=True)

                features[str(i).zfill(6)] = hist
                names_correspondence[str(i).zfill(6)] = image_name
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue

        new_df = pd.DataFrame(features)
        self.overall = pd.concat([self.overall, new_df], ignore_index=True)
        self.create_memo("Convert an image to G component, compute Local Binary Patterns, and then flatten them into a one-dimensional array.")
        self.create_names_correspondence(names_correspondence)


class LbpBlue(feather.Feather):
    """
    画像を読み込み、B成分に変換後、Local Binary Pattern 特徴量を計算する。

    Args:
        image_path (str): 画像のファイルパス。

    Returns:
        np.ndarray: LBP特徴量の一次元配列。

    Raises:
        IOError: 画像ファイルの読み込みに失敗した場合に発生。
    """
    def create_features(self):
        features = {}
        names_correspondence = {}

        for i in tqdm(range(len(self.image_paths)), desc="LBP for Blue"):
            path = self.image_paths[i]
            image_name = os.path.basename(path)

            try:
                # 画像を読み込み、グレースケールに変換
                image = self.load_image(path)
                converted_image = image[:, :, 0]

                # LBP特徴量を計算
                points, radius = 16, 2
                lbp = feature.local_binary_pattern(converted_image, points, radius, method="uniform")

                # ヒストグラムを計算
                hist, _ = np.histogram(lbp.ravel(), bins=points+2, range=(0, points+2), density=True)

                features[str(i).zfill(6)] = hist
                names_correspondence[str(i).zfill(6)] = image_name
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue

        new_df = pd.DataFrame(features)
        self.overall = pd.concat([self.overall, new_df], ignore_index=True)
        self.create_memo("Convert an image to B component, compute Local Binary Patterns, and then flatten them into a one-dimensional array.")
        self.create_names_correspondence(names_correspondence)


class LbpHue(feather.Feather):
    """
    画像を読み込み、H成分に変換後、Local Binary Pattern 特徴量を計算する。

    Args:
        image_path (str): 画像のファイルパス。

    Returns:
        np.ndarray: LBP特徴量の一次元配列。

    Raises:
        IOError: 画像ファイルの読み込みに失敗した場合に発生。
    """
    def create_features(self):
        features = {}
        names_correspondence = {}

        for i in tqdm(range(len(self.image_paths)), desc="LBP for Hue"):
            path = self.image_paths[i]
            image_name = os.path.basename(path)

            try:
                # 画像を読み込み、グレースケールに変換
                image = self.load_image(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                converted_image = image[:, :, 0]

                # LBP特徴量を計算
                points, radius = 16, 2
                lbp = feature.local_binary_pattern(converted_image, points, radius, method="uniform")

                # ヒストグラムを計算
                hist, _ = np.histogram(lbp.ravel(), bins=points+2, range=(0, points+2), density=True)

                features[str(i).zfill(6)] = hist
                names_correspondence[str(i).zfill(6)] = image_name
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue

        new_df = pd.DataFrame(features)
        self.overall = pd.concat([self.overall, new_df], ignore_index=True)
        self.create_memo("Convert an image to H component, compute Local Binary Patterns, and then flatten them into a one-dimensional array.")
        self.create_names_correspondence(names_correspondence)


class LbpSaturation(feather.Feather):
    """
    画像を読み込み、H成分に変換後、Local Binary Pattern 特徴量を計算する。

    Args:
        image_path (str): 画像のファイルパス。

    Returns:
        np.ndarray: LBP特徴量の一次元配列。

    Raises:
        IOError: 画像ファイルの読み込みに失敗した場合に発生。
    """
    def create_features(self):
        features = {}
        names_correspondence = {}

        for i in tqdm(range(len(self.image_paths)), desc="LBP for Saturation"):
            path = self.image_paths[i]
            image_name = os.path.basename(path)

            try:
                # 画像を読み込み、グレースケールに変換
                image = self.load_image(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                converted_image = image[:, :, 1]

                # LBP特徴量を計算
                points, radius = 16, 2
                lbp = feature.local_binary_pattern(converted_image, points, radius, method="uniform")

                # ヒストグラムを計算
                hist, _ = np.histogram(lbp.ravel(), bins=points+2, range=(0, points+2), density=True)

                features[str(i).zfill(6)] = hist
                names_correspondence[str(i).zfill(6)] = image_name
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue

        new_df = pd.DataFrame(features)
        self.overall = pd.concat([self.overall, new_df], ignore_index=True)
        self.create_memo("Convert an image to S component, compute Local Binary Patterns, and then flatten them into a one-dimensional array.")
        self.create_names_correspondence(names_correspondence)


class LbpValue(feather.Feather):
    """
    画像を読み込み、H成分に変換後、Local Binary Pattern 特徴量を計算する。

    Args:
        image_path (str): 画像のファイルパス。

    Returns:
        np.ndarray: LBP特徴量の一次元配列。

    Raises:
        IOError: 画像ファイルの読み込みに失敗した場合に発生。
    """
    def create_features(self):
        features = {}
        names_correspondence = {}

        for i in tqdm(range(len(self.image_paths)), desc="LBP for Value"):
            path = self.image_paths[i]
            image_name = os.path.basename(path)

            try:
                # 画像を読み込み、グレースケールに変換
                image = self.load_image(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                converted_image = image[:, :, 2]

                # LBP特徴量を計算
                points, radius = 16, 2
                lbp = feature.local_binary_pattern(converted_image, points, radius, method="uniform")

                # ヒストグラムを計算
                hist, _ = np.histogram(lbp.ravel(), bins=points+2, range=(0, points+2), density=True)

                features[str(i).zfill(6)] = hist
                names_correspondence[str(i).zfill(6)] = image_name
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue

        new_df = pd.DataFrame(features)
        self.overall = pd.concat([self.overall, new_df], ignore_index=True)
        self.create_memo("Convert an image to V component, compute Local Binary Patterns, and then flatten them into a one-dimensional array.")
        self.create_names_correspondence(names_correspondence)
