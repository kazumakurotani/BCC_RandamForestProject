import pandas as pd
import cv2

# データフレームを読み込む
dfs = pd.read_feather("feature\\overall\\labels_overall.ftr")

# データフレームをNumPy配列に変換
array = dfs.to_numpy()
print(array[0, :])
array = array[:, 50]

# 各行が128x128の画像データに対応していることを確認
expected_size = 128 * 128
if array.shape[0] != expected_size:
    raise ValueError(f"Each row of the input array must have a size of {expected_size}.")


image = array.reshape((128, 128))

# 画像を表示
cv2.imshow("test", image)

# キーボード入力を待つ
cv2.waitKey(0) & 0xFF == ord('q')

# すべてのウィンドウを閉じる
cv2.destroyAllWindows()
