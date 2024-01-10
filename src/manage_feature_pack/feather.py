import argparse
import csv
import inspect
import os
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path

import cv2
from tqdm import tqdm

import pandas as pd


def get_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in ({k: v for k, v in namespace.items()}).items():
        if inspect.isclass(v) and issubclass(v, Feather) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.overall_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


class Feather(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    overall_dir = 'feature\\overall'
    # train_dir = 'feature\\train'
    # test_dir = 'feature\\test'

    input_path = "data\\preprocessed"
    is_initialized = False  # 全サブクラス間で共有されるクラス変数
    image_paths = []
    label_names = []

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__).lstrip('_')
        self.overall = pd.DataFrame()
        # self.train = pd.DataFrame()
        # self.test = pd.DataFrame()
        self.overall_path = Path(self.overall_dir) / f'{self.name}.ftr'
        # self.train_path = Path(self.train_dir) / f'{self.name}_train.ftr'
        # self.test_path = Path(self.test_dir) / f'{self.name}_test.ftr'

        if not Feather.is_initialized:
            self._read_paths()
            Feather.is_initialized = True

    def run(self):
        self.create_features()
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        self.overall.columns = prefix + self.overall.columns + suffix
        # self.train.columns = prefix + self.train.columns + suffix
        # self.test.columns = prefix + self.test.columns + suffix
        return self

    def _read_paths(self) -> None:
        for dir in tqdm(os.listdir(self.input_path), desc="Reading Image Paths"):
            self.label_names.append(dir)
            dir_path = os.path.join(self.input_path, dir)
            for f in os.listdir(dir_path):
                file_path = os.path.join(dir_path, f)
                self.image_paths.append(file_path)

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        if os.path.isdir(self.overall_dir) is False:
            os.makedirs(self.overall_dir)
        self.overall.to_feather(str(self.overall_path))
        # self.train.to_feather(str(self.train_path))
        # self.test.to_feather(str(self.test_path))

    def load(self):
        self.overall = pd.read_feather(str(self.overall_path))
        # self.train = pd.read_feather(str(self.train_path))
        # self.test = pd.read_feather(str(self.test_path))

    # 特徴量メモをCSVファイルに残す
    def create_memo(self, desc):

        file_path = os.path.join(self.overall_dir, '_features_memo.csv')
        if not os.path.isfile(file_path):
            with open(file_path, "w"):
                pass

        with open(file_path, 'r+') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

            # 書き込まうとしている特徴量がすでに書き込まれていないかチェック
            col = [line for line in lines if line.split(',')[0] == self.name]
            if len(col) != 0:
                return

            writer = csv.writer(f)
            writer.writerow([self.name, desc])

    # ラベルと保存した名前の対応表をCSVファイルに残す
    def create_names_correspondence(self, names_correspondence):
        file_path = os.path.join(self.overall_dir, '_names_correspondence.csv')
        if not os.path.isfile(file_path):
            with open(file_path, "w"):
                pass

            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                # CSVのヘッダーを書き込む
                csv_writer.writerow(["Label", "Name"])
                # 辞書のデータを行としてCSVに書き込む
                for new_name, old_name in names_correspondence.items():
                    csv_writer.writerow([new_name, old_name])

        else:
            pass

    ############################################################

    def load_image(self, image_path):
        return cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
