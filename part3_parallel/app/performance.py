"""Сравнение последовательной и многопроцессорной обработки.

Запуск из папки part3_parallel:
    python -m app.performance --path ../data/input --output ../data/output/performance_report.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .augment import augment_image
from .producer_consumer import collect_image_files, run_parallel


def load_image(path, image_size=32):
    """Чтение изображения и нормализация в массив 0..1."""
    with Image.open(path) as image:
        converted = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(converted, dtype=float) / 255.0


def run_sequential(files):
    """Последовательная обработка: чтение и аугментация по одному файлу.

    Возвращает:
        число полученных массивов и время работы в секундах.
    """
    started = time.perf_counter()
    total = 0
    for path in files:
        array = load_image(path)
        total += len(augment_image(array))
    return total, time.perf_counter() - started


def run_parallel_version(files, workers=2):
    """Параллельная обработка через producer-consumer и пул процессов.

    Возвращает:
        число полученных массивов и время работы в секундах.
    """
    directory = files[0].parent if files else Path(".")
    results, elapsed = run_parallel(str(directory), workers=workers)
    total = sum(len(augmented) for _path, augmented in results)
    return total, elapsed


def build_report(files, output_file):
    """Измеряет обе версии и сохраняет отчёт сравнения в JSON-файл.

    Возвращает:
        словарь со статистикой сравнения.
    """
    sequential_total, sequential_time = run_sequential(files)
    parallel_total, parallel_time = run_parallel_version(files)
    report = {
        "files": len(files),
        "augmented_sequential": sequential_total,
        "augmented_parallel": parallel_total,
        "time_sequential_seconds": round(sequential_time, 4),
        "time_parallel_seconds": round(parallel_time, 4),
        "speedup": round(sequential_time / parallel_time, 2) if parallel_time else None,
    }
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def main(argv=None):
    """Точка входа: сравнение и запись performance_report.json."""
    parser = argparse.ArgumentParser(description="Замер производительности")
    parser.add_argument("--path", default="../data/input")
    parser.add_argument("--output", default="../data/output/performance_report.json")
    arguments = parser.parse_args(argv)
    files = collect_image_files(arguments.path)
    report = build_report(files, arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
