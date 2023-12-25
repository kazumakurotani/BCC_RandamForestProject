import os
import csv
import cv2
from src.manage_feature_pack import feather
import pandas as pd


def generate_features(self):
    feather.Feather.get_image_paths()
    args = feather.get_arguments()
    feather.generate_features(globals(), args.force)


class GrayscaleFeature(feather.Feather):
    def create_features(self):
        gray_image = self.load_image(image_path)
        features = self.extract_features_from_gray_image(gray_image)
            # 特徴量をDataFrameに追加する処理をここに記述します

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 以下のように実際の使用例を書くことができます
# if __name__ == "__main__":
#     grayscale_feature = GrayscaleFeature()
#     grayscale_feature.run().save()
