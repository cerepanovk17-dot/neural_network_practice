"""Паттерн «производитель-потребитель» с пулом процессов для аугментации.

Запуск из папки part3_parallel:
    python -m app.producer_consumer --path ../data/input --workers 2
"""

import argparse
import concurrent.futures
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .augment import augment_one

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
STOP_SIGNAL = None


def collect_image_files(root_dir):
    """Рекурсивный обход каталога и отбор файлов-изображений.

    Параметры:
        root_dir: исходная директория.
    Возвращает:
        отсортированный список путей pathlib.Path.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError("Каталог не найден: %s" % root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def producer(files, task_queue, consumers_count):
    """Поток-производитель: кладёт пути в очередь и посылает сигналы стопа.

    Параметры:
        files: список путей к файлам;
        task_queue: очередь queue.Queue с путями;
        consumers_count: число потоков-потребителей (по одному сигналу стопа каждому).
    """
    for path in files:
        task_queue.put(path)
    for _ in range(consumers_count):
        task_queue.put(STOP_SIGNAL)


def make_consumer(task_queue, loaded_queue, counter, counter_lock, logger_name):
    """Создаёт поток-потребитель: читает файл и кладёт массив в очередь результатов.

    Для каждого файла (включая нечитаемые) в loaded_queue кладётся ровно один
    элемент: кортеж (путь, массив) либо (путь, None) при ошибке. Это позволяет
    основному циклу остановиться без взаимной блокировки.

    Возвращает:
        функцию-потребителя для запуска в отдельном потоке.
    """
    def consumer():
        while True:
            path = task_queue.get()
            if path is STOP_SIGNAL:
                task_queue.task_done()
                break
            array = None
            try:
                with Image.open(path) as image:
                    converted = image.convert("RGB").resize((32, 32))
                    array = np.asarray(converted, dtype=float) / 255.0
                print("[%s] прочитан %s" % (logger_name, path.name))
            except (OSError, Image.DecompressionBombError) as error:
                print("[%s] файл %s пропущен: %s" % (logger_name, path.name, error))
            finally:
                loaded_queue.put((str(path), array))
                with counter_lock:
                    counter["done"] += 1
                task_queue.task_done()
    return consumer


def run_parallel(root_dir, workers=2, image_workers=2):
    """Параллельная обработка: потоки читают файлы, пул процессов аугментирует.

    Если пул процессов недоступен в среде исполнения (нет системных семафоров),
    аугментация выполняется в пуле потоков с предупреждением в консоли.

    Параметры:
        root_dir: каталог с изображениями;
        workers: число потоков-потребителей;
        image_workers: число процессов в пуле.
    Возвращает:
        список кортежей (путь, список аугментированных массивов) и время работы.
    """
    started = time.perf_counter()
    files = collect_image_files(root_dir)
    total_files = len(files)
    task_queue = queue.Queue()
    loaded_queue = queue.Queue()
    counter = {"done": 0}
    counter_lock = threading.Lock()

    producer_thread = threading.Thread(
        target=producer, args=(files, task_queue, workers), name="producer"
    )
    consumers = [
        threading.Thread(
            target=make_consumer(
                task_queue, loaded_queue, counter, counter_lock, "consumer-%d" % i
            ),
            name="consumer-%d" % i,
        )
        for i in range(workers)
    ]
    producer_thread.start()
    for thread in consumers:
        thread.start()

    use_processes = True
    try:
        pool = mp.Pool(processes=image_workers)
    except OSError as error:
        print("Пул процессов недоступен (%s): используется пул потоков." % error)
        pool = None
        use_processes = False

    results = []
    collected = 0
    if use_processes:
        async_results = []
        while collected < total_files:
            try:
                path, array = loaded_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            collected += 1
            if array is not None:
                async_results.append(pool.apply_async(augment_one, ((path, array),)))
        pool.close()
        pool.join()
        for result in async_results:
            path, augmented = result.get()
            results.append((path, augmented))
    else:
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=image_workers)
        pending = []
        while collected < total_files:
            try:
                path, array = loaded_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            collected += 1
            if array is not None:
                pending.append(thread_pool.submit(augment_one, (path, array)))
        for future in pending:
            path, augmented = future.result()
            results.append((path, augmented))
        thread_pool.shutdown(wait=True)

    for thread in consumers:
        thread.join()
    producer_thread.join()

    elapsed = time.perf_counter() - started
    return results, elapsed


def main(argv=None):
    """Точка входа демонстрации параллельной обработки.

    Параметры:
        argv: аргументы командной строки (для тестирования).
    """
    parser = argparse.ArgumentParser(description="Producer-consumer + пул процессов")
    parser.add_argument("--path", default="../data/input")
    parser.add_argument("--workers", type=int, default=2, help="Число потоков-потребителей")
    parser.add_argument("--image-workers", type=int, default=2, help="Число процессов")
    arguments = parser.parse_args(argv)
    results, elapsed = run_parallel(
        arguments.path, arguments.workers, arguments.image_workers
    )
    total_arrays = sum(len(augmented) for _path, augmented in results)
    print(
        "Файлов: %d, аугментированных массивов: %d, время: %.3f с"
        % (len(results), total_arrays, elapsed)
    )


if __name__ == "__main__":
    main()
