import cv2
import numpy as np
import random
import os


def generate_images(input_dir: str, output_dir: str, max_shift: int, rotation_step: int, n_rotations: int) -> None:
    """
    指定されたディレクトリ内の画像を処理し、結果を別のディレクトリに保存する。

    Args:
    input_dir (str): 入力画像のディレクトリ。
    output_dir (str): 出力画像のディレクトリ。
    max_shift (int): 画像移動の最大量。
    rotation_step (int): 回転のステップ（度）。
    n_rotations (int): 回転画像の数。

    Returns:
    None
    """
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = gi.load_image(image_path)

            # 画像の移動
            translated_image = gi.translate_image(image, max_shift)
            gi.save_image(translated_image, os.path.join(output_dir, f"translated_{filename}"))

            # 画像の回転
            rotated_images = gi.rotate_images(image, rotation_step, n_rotations)
            for i, rotated_image in enumerate(rotated_images):
                gi.save_image(rotated_image, os.path.join(output_dir, f"rotated_{i}_{filename}"))


def load_image(image_path: str) -> np.ndarray:
    """
    画像ファイルを読み込む。

    Args:
    image_path (str): 画像ファイルのパス。

    Returns:
    np.ndarray: 読み込まれた画像。
    """
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def translate_image(image: np.ndarray, max_shift: int) -> np.ndarray:
    """
    画像をランダムに移動させる。

    Args:
    image (np.ndarray): 入力画像。
    max_shift (int): 移動量の最大値。

    Returns:
    np.ndarray: 移動された画像。
    """
    rows, cols, _ = image.shape
    x_shift = random.randint(-max_shift, max_shift)
    y_shift = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(image, M, (cols, rows))

def rotate_images(image: np.ndarray, rotation_step: int, n_rotations: int) -> list:
    """
    画像を指定されたステップで複数回回転させる。

    Args:
    image (np.ndarray): 入力画像。
    rotation_step (int): 回転のステップ（度）。
    n_rotations (int): 生成する回転画像の数。

    Returns:
    list of np.ndarray: 回転された画像のリスト。
    """
    rows, cols, _ = image.shape
    rotated_images = []
    for i in range(n_rotations):
        M = cv2.getRotationMatrix2D((cols/2, rows/2), i * rotation_step, 1)
        rotated_images.append(cv2.warpAffine(image, M, (cols, rows)))
    return rotated_images

def save_image(image: np.ndarray, output_path: str) -> None:
    """
    画像をファイルに保存する。

    Args:
    image (np.ndarray): 保存する画像。
    output_path (str): 出力ファイルのパス。

    Returns:
    None
    """
    cv2.imwrite(output_path, image)
