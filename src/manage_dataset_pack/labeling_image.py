import os
import pandas as pd
from tqdm import tqdm
import shutil
import cv2

def labeling_image(class_index) -> None:
    """
    指定されたディレクトリ内の画像ファイルを読み込み、CSVファイルに基づいてラベルを確認し、
    画像ファイルの名前を変更して出力ディレクトリに保存する。

    Args:
        class_index: ラベリング項目

    Returns:
        None
    """
    # path
    input_image_dir_path = "data\\raw\\images"
    labels_file_path = "data\\raw\\labels\\labels.csv"
    labels_correspondence_file_path = "data\\raw\\labels\\labels_correspondence.csv"
    output_root_dir_path = "data\\raw\\labeled"
    output_logs_dir_path = "logs"

    # CSVファイルを読み込む
    labels_df = pd.read_csv(labels_file_path)
    labels_correspondence_df = pd.read_csv(labels_correspondence_file_path)

    # 出力ディレクトリの作成とパスの取得
    output_paths = _create_directories_from_csv(labels_correspondence_df, output_root_dir_path, class_index)

    # 画像の仕分け
    _sort_images(
        class_index,
        input_image_dir_path,
        output_root_dir_path,
        output_paths,
        labels_df,
        labels_correspondence_df
    )

    # 仕分け結果を記録
    result_text_file = os.path.join(output_logs_dir_path, "labeling.txt")
    _count_files(output_root_dir_path, result_text_file)

    _remove_empty_directories(output_root_dir_path, result_text_file)


def _create_directories_from_csv(database , output_root_path, class_index) -> None:
    """
    ラベリング結果を出力するディレクトリの作成

    Args:
        database (df): ラベル対応表
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
    for name in tqdm(database[class_index].unique(), desc="Creating Output Directories"):
        dir_path = os.path.join(output_root_path, name)
        output_paths[name] = dir_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok = True)

    return output_paths

def _sort_images(
        class_index,
        input_image_dir_path,
        output_root_dir_path,
        output_paths,
        labels_df,
        labels_correspondence_df):
    """
    ラベルごとに画像を選別する

    Args:
        class_index (str): 選別したいラベルの項目
        input_image_dir_path (str): 入力データがあるディレクトリパス
        output_paths (dict): 出力データの保存先となるディレクトリパス
        labels_df (df): ラベリング結果
        labels_correspondence_df (df): ラベルの対応表
    """
    for image_file in tqdm(os.listdir(input_image_dir_path), desc="Sorting Image"):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            labels = labels_df[labels_df['file'] == image_file].iloc[0]

            # どのラベルに対応するかの確認
            mapping = labels_correspondence_df[
                (labels_correspondence_df['type'] == labels['type']) &
                (labels_correspondence_df['name'] == labels['name']) &
                (labels_correspondence_df['state'] == labels['state'])
            ]

        # ラベリングされたデータか確認
        if not mapping.empty:
            class_name = mapping.iloc[0][class_index]
            # 新しいファイル名を生成
            new_filename = f"{class_name}_{image_file}"
            output_path = os.path.join(output_paths[class_name], new_filename)

            # 画像を出力ディレクトリに保存
            image_path = os.path.join(input_image_dir_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            cv2.imwrite(output_path, image)
        else:
            print(f"画像に想定していないデータが含まれています({image_file.name})")


def _count_files(output_root_dir_path: str, output_file: str) -> None:
    """
    指定されたディレクトリ内のファイル数をカウントし、
    結果をテキストファイルに保存する。

    Args:
    directory (str): ファイルをカウントするディレクトリのパス。
    output_file (str): 結果を保存するテキストファイルのパス。

    Returns:
    None
    """

    file_count = 0
    with open(output_file, "a") as f:
        # ファイルの初期化
        f.truncate(0)

        for class_index in os.listdir(output_root_dir_path):
            dir_path = os.path.join(output_root_dir_path, class_index)
            if os.path.isdir(dir_path):  # 現在の要素がディレクトリであることを確認します
                count = len(os.listdir(dir_path))  # ディレクトリ内のファイル数を数えます
                file_count += count
                f.write(f"{class_index}: {count}\n")

        f.write(f"合計ファイル数: {file_count}\n")


def _remove_empty_directories(output_root_dir_path: str, output_file: str) -> None:
    """
    指定されたディレクトリ内の空のディレクトリを削除する。

    Args:
    directory (str): 走査を開始するディレクトリのパス。

    Returns:
    None
    """

    with open(output_file, "a") as f:
        # 指定されたディレクトリ内のすべてのサブディレクトリを走査
        for root, dirs, files in os.walk(output_root_dir_path, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)

                # ディレクトリが空かどうかを確認
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    f.write(f"空のディレクトリを削除しました: {dir_path}\n")
