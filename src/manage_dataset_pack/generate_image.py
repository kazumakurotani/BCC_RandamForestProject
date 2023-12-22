import os
import random
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

import shutil


def generate_images(augmentation_option=0) -> None:
    """
    画像データを生成します。

    Args:
        augmentation_option (int): データ拡張のオプション。0または1。

    Returns:
        None
    """
    # parameters
    N = 8 # 画像の生成枚数
    SHIFT_REGION = 10 # shiftの移動領域

    # paths
    input_dir_path = "data\\raw\\labeled"
    output_dir_path = "data\\generated"
    output_log_dir_path = "logs"

    # 出力フォルダの作成
    _create_directories(input_dir_path, output_dir_path)

    # 画像データのパスの取得
    image_paths = _get_image_paths(input_dir_path)

    # 格納用変数
    generated_log = {}

    # 処理の開始
    for label_name, path_list in tqdm(image_paths.items(), desc="Generating images"):
        for image_path in path_list:
            # pathの生成
            image_path = os.path.join(input_dir_path, label_name, image_path)

            generated_images = {}
            log_list = []

            name_ext_pair = os.path.splitext(os.path.basename(image_path))

            # 画像を生成
            for i in range(N):
                num = str(i+1).zfill(2)  # 01, 02, ...

                # 画像の読み込み
                image = _load_image(image_path)

                if i == 0:
                    log = f"{num}, base"
                else:
                    if augmentation_option == 0:
                        image, param_rotate = _rotate_image(image, N, i)
                        image, param_shift = _shift_image(image, SHIFT_REGION)
                        log = f"{num}, rotate: {param_rotate}, shift: {param_shift}"
                    elif augmentation_option == 1:
                        image, param_rotate = _rotate_image(image, N, i)
                        log = f"{num}, rotate: {param_rotate}"

                new_filename = f"{name_ext_pair[0]}_{num}{name_ext_pair[1]}"

                generated_images[new_filename] = image
                log_list.append(log)

            # 画像の保存
            for file_name, generated_image in generated_images.items():
                _save_image(generated_image, output_dir_path, label_name, file_name)

            # logの追記
            generated_log[name_ext_pair[0]] = log_list

    _save_log(output_log_dir_path, generated_log)


def _create_directories(input_dir_path: str, output_root_path: str) -> None:
    """
    ラベリング結果を出力するディレクトリの作成

    Args:
        output_root_path (str): 保存先のルートパス
        class_index (str): ラベリングしたい項目

    Returns:
        output_paths(dict): 出力先のパス
    """
    # 格納用変数
    output_paths = {}

    # ディレクトリ内のデータを全削除
    shutil.rmtree(output_root_path)

    # 各ユニークな値に対してディレクトリを作成
    for name in tqdm(os.listdir(input_dir_path), desc="Creating Output Directories"):
        dir_path = os.path.join(output_root_path, name)
        output_paths[name] = dir_path
        os.makedirs(dir_path, exist_ok = True)


def _get_image_paths(input_dir_path: str) -> Dict[str, List[str]]:
    """
    与えられたディレクトリ内の画像ファイルのパスをラベル別に収集します。

    Args:
        input_dir_path (str): 画像ファイルが格納されているディレクトリのパス。

    Returns:
        Dict[str, List[str]]: ラベルごとの画像ファイルのパスを含むディクショナリ。
            キーはラベルディレクトリの名前で、値はそのラベルに関連付けられた画像ファイルのパスのリストです。
    """
    # 格納用変数
    image_paths = {}

    # datasetのパスを取得
    for label_dir in os.listdir(input_dir_path):
        label_dir_path = os.path.join(input_dir_path, label_dir)

        label_dir_image_paths = [
            f for f in os.listdir(label_dir_path) if os.path.isdir(label_dir_path)
        ]

        image_paths[label_dir] = label_dir_image_paths

    return image_paths


def _load_image(image_path: str) -> np.ndarray:
    """
    画像ファイルを指定されたパスから読み込みます。

    Args:
        image_path (str): 読み込む画像ファイルのパス。

    Returns:
        np.ndarray: 読み込まれた画像データのNumPy配列。

    Raises:
        ValueError: サポートされていないファイル形式の場合に発生します。
    """
    # 画像ファイル形式をチェック
    supported_formats = ('.png', '.jpg', '.jpeg')
    if image_path.lower().endswith(supported_formats):
        # OpenCVを使用して画像を読み込む
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{image_path}: 画像を読み込めませんでした。")
        return image
    else:
        raise ValueError(f"{image_path}: サポートされていないデータ形式です。")


def _shift_image(image: np.ndarray, SHIFT_REGION) -> np.ndarray:
    """
    画像をランダムにシフトする。

    Args:
        image (np.ndarray): シフトする画像。

    Returns:
        np.ndarray: シフトされた画像。
    """
    # シフト量の決定
    shift_x = random.randint(-SHIFT_REGION, SHIFT_REGION)
    shift_y = random.randint(-SHIFT_REGION, SHIFT_REGION)

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # シフトの実行
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 記載用変数の生成
    param = f"{shift_x}, {shift_y}"

    return shifted, param


def _rotate_image(image: np.ndarray, N, i) -> np.ndarray:
    """
    与えられた画像をランダムな角度で回転させる。

    Args:
        image (np.ndarray): 回転させる画像。形状は (高さ, 幅, チャネル) の形式。

    Returns:
        np.ndarray: 回転後の画像。
    """
    # 角度をランダムに生成
    angle = 360/N * i

    # 回転行列の生成
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)

    # 回転の実行
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 記載用変数の生成
    param = f"{angle}"

    return rotated, param


def _save_image(image: np.ndarray, output_dir_path: str, label_name: str, new_filename: str) -> None:
    """
    画像を指定されたディレクトリに保存します。

    Args:
        image (np.ndarray): 保存する画像データのNumPy配列。
        output_dir_path (str): 保存先のディレクトリのパス。
        label_name (str): 画像のラベル名。
        new_filename (str): 新しいファイル名。

    Returns:
        None
    """
    output_path = os.path.join(output_dir_path, label_name, new_filename)
    cv2.imwrite(output_path, image)


def _save_log(output_log_dir_path: str, generated_log: Dict[str, List[str]]) -> None:
    """
    生成されたログを指定されたファイルに保存します。

    Args:
        output_dir_path (str): ログファイルの保存先ディレクトリのパス。
        generated_log (Dict[str, List[str]]): 生成されたログ情報の辞書。

    Returns:
        None
    """
    # outputpathの生成
    output_file = os.path.join(output_log_dir_path, "generating.txt")

    with open(output_file, "a") as f:
        # ファイルの初期化
        f.truncate(0)
        for base_file_name, log_list in generated_log.items():
            for log in log_list:
                f.write(f"{base_file_name}, {log}\n")
