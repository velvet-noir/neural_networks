import os

import numpy as np


def load_drawings(folder="drawings"):
    X = []  # список изображений (flatten)
    y = []  # список меток ('o', 'x', 'z')

    for filename in os.listdir(folder):
        if not filename.endswith(".npy"):
            continue

        path = os.path.join(folder, filename)
        data = np.load(path)

        # нормализуем и делаем вектор
        X.append(data.flatten().astype(float))

        # метка по первому символу файла
        label = filename[0].lower()
        if label not in ["o", "x", "z"]:
            label = "?"
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"✅ Загружено {len(X)} рисунков из папки '{folder}'")

    return X, y


if __name__ == "__main__":
    X, y = load_drawings("lab_2/drawings")
    print("Размер X:", X.shape)
    print("Метки классов:", np.unique(y, return_counts=True))
