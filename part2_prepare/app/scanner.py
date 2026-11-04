"""Обход каталога и преобразование файлов в единый массив NumPy."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def collect_files(root_dir, extensions):
    """Рекурсивный обход каталога с отбором файлов по расширениям.

    Параметры:
        root_dir: исходная директория (str или pathlib.Path);
        extensions: итерируемое расширений без точки, например ['png', 'csv'].
    Возвращает:
        отсортированный список путей pathlib.Path.
    """
    allowed = {"." + item.lower().lstrip(".") for item in extensions}
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError("Каталог не найден: %s" % root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )


def load_image_array(path, image_size):
    """Чтение изображения через Pillow, приведение к одному размеру и нормализация.

    Параметры:
        path: путь к изображению;
        image_size: целевой размер стороны (изображение приводится к квадрату).
    Возвращает:
        одномерный массив float в диапазоне [0, 1].
    """
    with Image.open(path) as image:
        converted = image.convert("RGB").resize(
            (image_size, image_size), Image.BILINEAR
        )
        array = np.asarray(converted, dtype=float) / 255.0
    return array.reshape(-1)


def load_csv_array(path):
    """Чтение CSV как таблицы: числовые столбцы в одномерный массив."""
    table = pd.read_csv(path, sep=None, engine="python")
    numeric = table.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("В файле %s нет числовых столбцов." % path)
    return numeric.to_numpy(dtype=float).reshape(-1)


def prepare_dataset(root_dir, extensions, output_file, image_size=32):
    """Сбор всех файлов каталога в один массив и сохранение в .npy.

    Параметры:
        root_dir: исходная директория;
        extensions: список расширений;
        output_file: путь к итоговому файлу .npy;
        image_size: сторона изображения при приведении.
    Возвращает:
        словарь со статистикой обработки.
    """
    started = time.perf_counter()
    files = collect_files(root_dir, extensions)
    arrays = []
    counted = {"images": 0, "csv": 0, "failed": 0}
    for path in files:
        try:
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                arrays.append(load_image_array(path, image_size))
                counted["images"] += 1
            else:
                arrays.append(load_csv_array(path))
                counted["csv"] += 1
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            counted["failed"] += 1
            print("Пропущен файл %s: %s" % (path, error))
    if not arrays:
        raise ValueError("Не найдено ни одного подходящего файла.")
    max_length = max(len(row) for row in arrays)
    padded = np.zeros((len(arrays), max_length), dtype=float)
    for index, row in enumerate(arrays):
        padded[index, : len(row)] = row
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, padded)
    elapsed = time.perf_counter() - started
    stats = {
        "files_total": len(files),
        "images": counted["images"],
        "csv": counted["csv"],
        "failed": counted["failed"],
        "matrix_shape": list(padded.shape),
        "output_file": str(output_path),
        "elapsed_seconds": round(elapsed, 3),
    }
    return stats
