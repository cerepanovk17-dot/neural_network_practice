"""Чтение настроек из переменных окружения с резервными значениями."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv опциональна
    pass


def setting(name, default):
    """Возвращает значение переменной окружения или значение по умолчанию."""
    return os.getenv(name, default)


DATA_DIR = setting("DATA_DIR", "data/input")
OUTPUT_DIR = setting("OUTPUT_DIR", "data/output")
LOG_LEVEL = setting("LOG_LEVEL", "INFO")
LOG_FILE = setting("LOG_FILE", "logs/app.log")
IMAGE_SIZE = int(setting("IMAGE_SIZE", "32"))
