import cv2
import shutil
import os
from tqdm import tqdm
from typing import Dict, List


def preprocess() -> None:
    """
    画像に対してガウシアンブラーによる平滑化とリサイズを行う。

    ガウシアンフィルタを適用して画像を平滑化し、指定されたサイズにリサイズする。

    Args:
        image (np.ndarray): 処理する画像。BGRカラースペースであることが想定される。

    Returns:
        np.ndarray: 平滑化およびリサイズされた画像。
    """
    output_root_path = "data\\preprocessed"
    input_root_path = "data\\generated"

    _create_output_directories(output_root_path, input_root_path)
    input_paths = _get_image_paths(input_root_path)

    # 処理の開始
    for label_name, path_list in tqdm(input_paths.items(), desc="Proprocessing images"):
        for image_path in path_list:
            # pathの生成
            input_path = os.path.join(input_root_path, label_name, image_path)
            output_path = os.path.join(output_root_path, label_name, image_path)

                # 画像の読み込み
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)

            # パラメータの設定
            size_resize = (128, 128)
            kernel_size_smooth = (3, 3)
            sigma_smooth = 1

            # 画像の前処理: ガウシアンブラーによる平滑化、リサイズ
            smoothed_image = cv2.GaussianBlur(image, kernel_size_smooth, sigma_smooth)
            resized_image = cv2.resize(smoothed_image, size_resize)

            cv2.imwrite(output_path, resized_image)

    _remove_empty_directories(output_root_path)

def _create_output_directories(output_root_path, input_path):
    # ディレクトリ内のデータを全削除
    shutil.rmtree(output_root_path)

    for name in tqdm(os.listdir(input_path), desc="Creating Output Directories"):
        dir_path = os.path.join(output_root_path, name)
        if not os.path.exists(dir_path):
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
    extracted_cells = {
        "多染性赤芽球": "Polychromatic erythroblast",
        "リンパ球": "Lymphocyte",
        "骨髄球": "Myelocyte",
        "後骨髄球": "Metamyelocyte",
        "桿状核球": "Stab cell",
        "分葉核球": "Segmented cell",
        "前骨髄球": "Premyelocyte",
        "好酸球": "Eosinophil",
        "正染性赤芽球": "Orthochromatic erythroblast",
        "塩基性赤芽球": "Basophilic erythroblast"
    }

    # 格納用変数
    image_paths = {}

    # datasetのパスを取得
    for label_dir in extracted_cells.values():
        label_dir_path = os.path.join(input_dir_path, label_dir)

        label_dir_image_paths = [
            f for f in os.listdir(label_dir_path) if os.path.isdir(label_dir_path)
        ]

        image_paths[label_dir] = label_dir_image_paths

    return image_paths

def _remove_empty_directories(output_root_dir_path: str) -> None:
    """
    指定されたディレクトリ内の空のディレクトリを削除する。

    Args:
    directory (str): 走査を開始するディレクトリのパス。

    Returns:
    None
    """
    for root, dirs, files in os.walk(output_root_dir_path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)

            # ディレクトリが空かどうかを確認
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
