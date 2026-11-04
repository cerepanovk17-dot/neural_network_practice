"""Консольная утилита препроцессинга: команды prepare и doctor.

Запуск из папки part2_prepare:
    python -m app.cli prepare --path ../data/input --ext png,csv --output ../data/output/dataset.npy
    python -m app.cli doctor
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .config import DATA_DIR, IMAGE_SIZE, OUTPUT_DIR
from .logging_config import setup_logging
from .scanner import prepare_dataset


def command_prepare(arguments, logger):
    """Команда prepare: сбор данных из каталога в единый .npy-массив."""
    logger.info("Старт prepare: %s", arguments.path)
    stats = prepare_dataset(
        arguments.path,
        [item.strip() for item in arguments.ext.split(",")],
        arguments.output,
        image_size=arguments.image_size,
    )
    for key, value in stats.items():
        print("%s: %s" % (key, value))
    logger.info("Готово: %s", stats["output_file"])


def command_doctor(arguments, logger):
    """Команда doctor: проверка окружения и внешних инструментов."""
    print("Python: %s" % platform.python_version())
    print("Рабочая директория: %s" % Path.cwd())
    for module in ("numpy", "PIL", "pandas", "matplotlib", "dotenv"):
        try:
            imported = __import__(module)
            print("Библиотека %s: %s" % (module, getattr(imported, "__version__", "есть")))
        except ImportError:
            print("Библиотека %s: НЕ НАЙДЕНА" % module)
    for tool in ("nvidia-smi", "git"):
        found = shutil.which(tool)
        if found:
            try:
                result = subprocess.run(
                    [tool], capture_output=True, text=True, timeout=10
                )
                summary = (result.stdout or result.stderr).strip().splitlines()[0]
                print("%s: %s" % (tool, summary))
            except (subprocess.SubprocessError, OSError) as error:
                print("%s: ошибка вызова (%s)" % (tool, error))
        else:
            print("%s: не найден в PATH (CUDA, вероятно, отсутствует)" % tool)
    logger.info("Проверка окружения завершена.")


def build_parser():
    """Создаёт парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="prepare-tool", description="Утилита подготовки данных"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Собрать файлы в .npy")
    prepare.add_argument("--path", default=DATA_DIR, help="Каталог с данными")
    prepare.add_argument("--ext", default="png,jpg,csv", help="Расширения через запятую")
    prepare.add_argument("--output", default=str(Path(OUTPUT_DIR) / "dataset.npy"))
    prepare.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    prepare.set_defaults(handler=command_prepare)

    doctor = subparsers.add_parser("doctor", help="Проверка окружения")
    doctor.set_defaults(handler=command_doctor)
    return parser


def main(argv=None):
    """Точка входа CLI."""
    logger = setup_logging()
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        arguments.handler(arguments, logger)
    except (OSError, ValueError, NotADirectoryError) as error:
        logger.error("Ошибка: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
