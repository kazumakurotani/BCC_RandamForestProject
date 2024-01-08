import os
# import shutil
from abc import ABCMeta
import cv2
from tqdm import tqdm


class ImagePaths(metaclass=ABCMeta):
    image_paths = []
    input_path = "data\\generated"

    is_initialized = False

    def __init__(self):
        if self.is_initialized is False:
            self._read_paths()
            self.is_initialized = True
        else:
            pass

    def _read_paths(self) -> None:
        for dir in tqdm(os.listdir(self.input_path), desc="Reading Image Paths"):
            dir_path = os.path.join(self.input_path, dir)
            for f in os.listdir(dir_path):
                file_path = os.path.join(dir_path, f)
                self.image_paths.append(file_path)

    def load_image(self, image_path):
        return cv2.imread(image_path, cv2.IMREAD_COLOR)
