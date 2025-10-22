import os
import tkinter as tk

import numpy as np


class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Рисовалка для сети Кохонена")

        # параметры холста
        self.grid_size = 50  # 50x50 матрица
        self.canvas_size = 500  # пиксельный размер холста
        self.pixel_size = self.canvas_size // self.grid_size

        # создаем холст
        self.canvas = tk.Canvas(
            root, width=self.canvas_size, height=self.canvas_size, bg="black"
        )
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)

        # обработчик рисования
        self.canvas.bind("<B1-Motion>", self.paint)

        # матрица пикселей (0 — черный, 1 — белый)
        self.data = np.zeros((self.grid_size, self.grid_size), dtype=float)

        # панель кнопок
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Button(btn_frame, text="Очистить", command=self.clear_canvas).pack(
            fill=tk.X, pady=5
        )
        tk.Button(btn_frame, text="Сохранить", command=self.save_image).pack(
            fill=tk.X, pady=5
        )

        # создаем папку для сохранения данных
        os.makedirs("lab_2/drawings", exist_ok=True)

    # рисование при движении мыши
    def paint(self, event):
        x, y = event.x // self.pixel_size, event.y // self.pixel_size
        for dx in [0, 1]:
            for dy in [0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.data[ny, nx] = 1.0
                    self.draw_pixel(nx, ny, 1)

    # отрисовка одного квадрата
    def draw_pixel(self, x, y, value):
        x1, y1 = x * self.pixel_size, y * self.pixel_size
        x2, y2 = x1 + self.pixel_size, y1 + self.pixel_size
        color = "white" if value == 1 else "black"
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

    # очистка холста
    def clear_canvas(self):
        self.canvas.delete("all")
        self.data.fill(0)

    # сохранение нарисованного изображения
    def save_image(self):
        # формируем имя файла
        existing = [f for f in os.listdir("lab_2/drawings") if f.startswith("drawing_")]
        next_id = len(existing) + 1
        filename = f"lab_2/drawings/drawing_{next_id}.npy"

        np.save(filename, self.data)
        print(f"✅ Изображение сохранено: {filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
