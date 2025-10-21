import random
import tkinter as tk

import numpy as np
from nn import NeuralNetwork


class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обратное распространение")

        self.canvas_size = 280
        self.grid_size = 28
        self.pixel_size = self.canvas_size // self.grid_size

        # холст для рисования
        self.canvas = tk.Canvas(
            root, width=self.canvas_size, height=self.canvas_size, bg="black"
        )
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.data = np.zeros((self.grid_size, self.grid_size), dtype=float)

        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Button(btn_frame, text="Очистить", command=self.clear_canvas).pack(
            fill=tk.X, pady=5
        )
        tk.Button(btn_frame, text="Определить", command=self.detect).pack(
            fill=tk.X, pady=5
        )
        tk.Button(btn_frame, text="Тест", command=self.test_random_image).pack(
            fill=tk.X, pady=5
        )

        self.nn = NeuralNetwork(784, 32, 10)
        weights = np.load("backpropagation_nn/mnist_trained_weights.npz")
        self.nn.W1 = weights["W1"]
        self.nn.b1 = weights["b1"]
        self.nn.W2 = weights["W2"]
        self.nn.b2 = weights["b2"]

    def paint(self, event):
        x, y = event.x // self.pixel_size, event.y // self.pixel_size

        for dx in [0, 1]:
            for dy in [0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.data[ny, nx] = 1.0
                    self.draw_pixel(nx, ny, 1)

    def draw_pixel(self, x, y, value):
        x1, y1 = x * self.pixel_size, y * self.pixel_size
        x2, y2 = x1 + self.pixel_size, y1 + self.pixel_size
        color = "white" if value == 1 else "black"
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.data.fill(0)

    def detect(self):
        input_vector = self.data.flatten().astype(float)

        output, _ = self.nn.forward(input_vector)

        predicted = np.argmax(output)
        print(f"Предсказание сети: {predicted}")

    def test_random_image(self):
        data = np.load("data/mnist_data_ready.npz")
        X_test = data["X_test"]
        y_test = data["y_test"]

        index = random.randint(0, len(X_test) - 1)
        image = X_test[index].reshape(28, 28)
        label_vector = y_test[index]
        true_label = np.argmax(label_vector)

        self.clear_canvas()
        binary_image = (image > 0.5).astype(float)
        self.data = binary_image
        for y in range(28):
            for x in range(28):
                if self.data[y, x] > 0.5:
                    self.draw_pixel(x, y, 1)

        input_vector = binary_image.flatten()

        output, _ = self.nn.forward(input_vector)

        predicted_label = np.argmax(output)

        print(
            f"Правильный ответ: {true_label} || Предсказание сети: {predicted_label}\n"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
