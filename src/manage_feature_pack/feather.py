import argparse
import csv
import inspect
import os
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path

from typing import List

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
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


class Feather(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    overall_dir = 'feature\\overall'
    train_dir = 'feature\\train'
    test_dir = 'feature\\test'
    image_paths = []
    directory_path = 

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__).lstrip('_')
        self.overall = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.overall_path = Path(self.overall_dir) / f'{self.name}_train.ftr'
        self.train_path = Path(self.train_dir) / f'{self.name}_train.ftr'
        self.test_path = Path(self.test_dir) / f'{self.name}_test.ftr'

    def run(self):
        self.create_features()
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        self.overall.columns = prefix + self.overall.columns + suffix
        self.train.columns = prefix + self.train.columns + suffix
        self.test.columns = prefix + self.test.columns + suffix
        return self

    def get_image_paths(self) -> List[str]:
        """
        指定されたディレクトリ内のファイルのリストを返す。

        Args:
            directory_path (str): 読み込むディレクトリのパス。

        Returns:
            List[str]: ディレクトリ内のファイルのパスのリスト。
        """
        # datasetのパスを取得
        for label_dir in os.listdir(directory_path):
            label_dir_path = os.path.join(directory_path, label_dir)
            for file in os.listdir(label_dir_path):
                file_path = os.path.join(label_dir_path, file)
                self.image_paths.append(file_path)

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.overall.to_feather(str(self.overall_path))
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))

    def load(self):
        self.overall = pd.read_feather(str(self.overall_path))
        self.train = pd.read_feather(str(self.train_path))
        self.test = pd.read_feather(str(self.test_path))

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
