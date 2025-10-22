import numpy as np
import random
from load_drawings import load_drawings   # импортируй свою функцию из предыдущего файла

def check_dataset(folder="lab_2/drawings"):
    X, y = load_drawings(folder)

    print("\n=== Проверка данных ===")
    print(f"Форма X: {X.shape}  (n образов, m признаков)")
    print(f"Форма y: {y.shape}")
    print(f"Классы: {np.unique(y)}")
    print("Количество в каждом классе:", dict(zip(*np.unique(y, return_counts=True))))
    print(f"Диапазон значений: min={X.min():.2f}, max={X.max():.2f}")
    print(f"Среднее значение по всем пикселям: {X.mean():.4f}")
    print(f"Доля ненулевых пикселей: {(np.count_nonzero(X) / X.size):.4%}")

    # Выводим один случайный образ
    idx = random.randint(0, len(X) - 1)
    label = y[idx]
    image = X[idx].reshape(50, 50)

    print(f"\nПример образа #{idx} (метка '{label}'):")
    # выводим в консоль "псевдоизображение"
    for row in image:
        line = "".join("█" if px > 0.5 else " " for px in row)
        print(line)

    print("\n✅ Проверка завершена.")


if __name__ == "__main__":
    check_dataset()
