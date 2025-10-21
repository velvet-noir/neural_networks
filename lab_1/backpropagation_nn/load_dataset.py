import numpy as np
from sklearn.datasets import fetch_openml


def download_and_save_mnist(save_path="lab_1/data/mnist_data.npz"):
    print("Загрузка MNIST с OpenML...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")

    X = mnist.data.astype(np.uint8)
    y = mnist.target.astype(np.uint8)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    np.savez_compressed(
        save_path,
        X_train=X_train.reshape(-1, 28, 28),
        y_train=y_train,
        X_test=X_test.reshape(-1, 28, 28),
        y_test=y_test,
    )

    print(f"MNIST успешно загружен и сохранён в {save_path}")


def normalize_mnist(
    npz_path="lab_1/data/mnist_data.npz", save_path="lab_1/data/mnist_data_normalized.npz"
):
    data = np.load(npz_path)
    X_train = data["X_train"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_train = data["y_train"]
    y_test = data["y_test"]

    X_train /= 255.0
    X_test /= 255.0

    np.savez_compressed(
        save_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    print(f"Данные MNIST нормализованы и сохранены в {save_path}")


def one_hot_encode_mnist(
    npz_path="lab_1/data/mnist_data_normalized.npz",
    save_path="lab_1/data/mnist_data_ready.npz",
    num_classes=10,
):
    data = np.load(npz_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    def one_hot(y, num_classes):
        res = np.zeros((y.size, num_classes))
        res[np.arange(y.size), y] = 1
        return res

    y_train_oh = one_hot(y_train, num_classes)
    y_test_oh = one_hot(y_test, num_classes)

    np.savez_compressed(
        save_path, X_train=X_train, y_train=y_train_oh, X_test=X_test, y_test=y_test_oh
    )

    print(f"Данные MNIST с one-hot метками сохранены в {save_path}")


def show_mnist_sample(npz_path="lab_1/data/mnist_data_ready.npz", index=0):
    data = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]

    image = X_train[index]
    label_one_hot = y_train[index]

    print(f"One-hot вектор метки: {label_one_hot}")
    print("Изображение (матрица 28x28):")
    print(image)


def check_mnist_shapes(npz_path="lab_1/data/mnist_data_mini.npz"):
    data = np.load(npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"Количество образов в обучающем наборе: {X_train.shape[0]}")
    print(f"Количество образов в тестовом наборе: {X_test.shape[0]}")
    print(f"Размер каждого изображения: {X_train.shape[1:]}")
    print(f"Размер one-hot меток: {y_train.shape[1]}")


def create_mini_mnist(
    npz_path="lab_1/data/mnist_data_ready.npz",
    save_path="lab_1/data/mnist_data_mini.npz",
    train_samples_per_class=200,
    test_samples=200,
    num_classes=10,
):
    data = np.load(npz_path)
    X_train_full, y_train_full = data["X_train"], data["y_train"]
    X_test_full, y_test_full = data["X_test"], data["y_test"]

    X_train_mini = []
    y_train_mini = []

    for c in range(num_classes):
        indices = np.where(y_train_full[:, c] == 1)[0]
        selected = np.random.choice(indices, train_samples_per_class, replace=False)
        X_train_mini.append(X_train_full[selected])
        y_train_mini.append(y_train_full[selected])

    X_train_mini = np.vstack(X_train_mini)
    y_train_mini = np.vstack(y_train_mini)

    total_test = X_test_full.shape[0]
    test_indices = np.random.choice(total_test, test_samples, replace=False)
    X_test_mini = X_test_full[test_indices]
    y_test_mini = y_test_full[test_indices]

    np.savez_compressed(
        save_path,
        X_train=X_train_mini,
        y_train=y_train_mini,
        X_test=X_test_mini,
        y_test=y_test_mini,
    )

    print(f"Мини-датасет создан и сохранён в {save_path}")
    print(f"Размер мини-обучающего набора: {X_train_mini.shape[0]}")
    print(f"Размер мини-тестового набора: {X_test_mini.shape[0]}")


if __name__ == "__main__":
    # download_and_save_mnist()
    # normalize_mnist()
    # one_hot_encode_mnist()
    # show_mnist_sample()
    # check_mnist_shapes()
    create_mini_mnist()
