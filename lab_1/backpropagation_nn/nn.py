import numpy as np


def load_mnist_mini(file_path="data/mnist_data_mini.npz"):
    data = np.load(file_path)
    X_train = data["X_train"]
    y_train = data["y_train"]

    X_train = X_train.reshape(X_train.shape[0], -1)

    training_data = [(X_train[i], y_train[i]) for i in range(X_train.shape[0])]
    return training_data


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Инициализация весов и смещений нейросети
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))

        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        """
        Прямой проход для одного образа
        x: входной вектор shape (784,)
        Возвращает:
            - выход сети (output)
            - кэш с промежуточными значениями для обратного прохода
        """
        x = x.reshape(-1, 1)

        # Скрытый слой
        z1 = np.dot(self.W1, x) + self.b1
        a1 = self.sigmoid(z1)

        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.sigmoid(z2)
        cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

        return a2, cache

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, cache, target_vector, learning_rate):
        """
        Обратный проход и обновление весов по одному образу
        cache: словарь с промежуточными значениями из forward
        target_vector: true target vector shape (output_size, 1)
        learning_rate: скорость обучения
        """
        x = cache["x"]  # (input_size, 1)
        a1 = cache["a1"]  # (hidden_size, 1)
        a2 = cache["a2"]  # (output_size, 1)

        y = target_vector.reshape(-1, 1)

        delta2 = (a2 - y) * self.sigmoid_derivative(a2)
        dW2 = np.dot(delta2, a1.T)
        db2 = delta2

        delta1 = np.dot(self.W2.T, delta2) * self.sigmoid_derivative(a1)
        dW1 = np.dot(delta1, x.T)
        db1 = delta1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(
        self,
        training_data,
        learning_rate,
        epochs,
        epsilon,
        test_data=None,
        test_interval=50,
    ):
        """
        Обучение сети по одному образу
        training_data: список кортежей (input_vector, target_vector)
        learning_rate: начальная скорость обучения
        epochs: максимальное количество эпох
        epsilon: порог ошибки для досрочной остановки
        """
        MIN_LEARNING_RATE = 0.0001

        for epoch in range(1, epochs + 1):
            np.random.shuffle(training_data)
            all_below_epsilon = True
            total_loss = 0

            count_below_epsilon = 0

            for x, y in training_data:
                output, cache = self.forward(x)

                y_col = y.reshape(-1, 1)
                loss = self.mse_loss(y_col, output)
                total_loss += loss

                if loss < epsilon:
                    count_below_epsilon += 1
                else:
                    all_below_epsilon = False

                self.backward(cache, y, learning_rate)

            avg_loss = total_loss / len(training_data)

            if avg_loss < 0.01:
                learning_rate = max(learning_rate * 0.95, MIN_LEARNING_RATE)
            elif avg_loss < 0.005:
                learning_rate = max(learning_rate * 0.999, MIN_LEARNING_RATE)

            print(
                f"Эпоха {epoch}/{epochs} - средняя MSE: {avg_loss:.6f} - learning_rate: {learning_rate:.5f}"
            )

            if test_data is not None and epoch % test_interval == 0:
                results, accuracy = self.test_random_samples(test_data, num_samples=200)
                print(
                    f"Промежуточный тест после эпохи {epoch}: точность {accuracy:.2f}%"
                )

            print(
                f"Эпоха {epoch}: {count_below_epsilon}/{len(training_data)} образов имеют ошибку < EPSILON"
            )
            # Досрочная остановка
            if all_below_epsilon:
                print(
                    f"Обучение остановлено досрочно на эпохе {epoch} — все ошибки < {epsilon}"
                )
                break

    def save_weights(self, file_path="trained_weights.npz"):
        """
        Сохраняет веса и смещения сети в файл .npz
        """
        np.savez_compressed(file_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Весы и смещения сохранены в {file_path}")

    def test_random_samples(self, test_data, num_samples=1000):
        """
        Тестирует сеть на нескольких случайных образах из тестового набора
        test_data: список кортежей (input_vector, target_vector)
        num_samples: количество случайных образов для теста
        """
        num_samples = min(num_samples, len(test_data))
        indices = np.random.choice(len(test_data), num_samples, replace=False)

        correct = 0
        results = []

        for idx in indices:
            input_vector, target_vector = test_data[idx]
            output, _ = self.forward(input_vector)
            predicted_class = np.argmax(output)
            true_class = np.argmax(target_vector)

            results.append((predicted_class, true_class))

            if predicted_class == true_class:
                correct += 1

        accuracy = correct / num_samples * 100

        print(f"\n=== Тест {num_samples} случайных образов ===")
        print(f"Правильных предсказаний: {correct}/{num_samples} ({accuracy:.2f}%)")

        return results, accuracy


def load_mnist_mini_test(file_path="data/mnist_data_ready.npz"):
    """
    Загружает тестовую выборку мини-MNIST
    """
    data = np.load(file_path)
    X_test = data["X_test"]
    y_test = data["y_test"]

    X_test = X_test.reshape(X_test.shape[0], -1)

    test_data = [(X_test[i], y_test[i]) for i in range(X_test.shape[0])]
    return test_data


if __name__ == "__main__":
    INPUT_SIZE = 784
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 10
    LEARNING_RATE = 0.3
    EPOCHS = 200
    EPSILON = 0.1
    TEST_INTERVAL = 10

    training_data = load_mnist_mini()
    test_data = load_mnist_mini_test()

    nn = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    nn.train(
        training_data,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        epsilon=EPSILON,
        test_data=test_data,
        test_interval=TEST_INTERVAL,
    )
    nn.save_weights("backpropagation_nn/mnist_trained_weights.npz")
    print("Обучение завершено и веса сохранены.")

    # weights = np.load("backpropagation_nn/mnist_trained_weights.npz")
    # nn.W1 = weights["W1"]
    # nn.b1 = weights["b1"]
    # nn.W2 = weights["W2"]
    # nn.b2 = weights["b2"]
    # nn.test_random_samples(test_data)
