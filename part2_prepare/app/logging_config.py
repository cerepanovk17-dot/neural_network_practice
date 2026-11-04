"""Настройка журналирования приложения."""

import logging
from pathlib import Path

from .config import LOG_FILE, LOG_LEVEL

# Корень проекта: на два уровня выше папки app
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(level=None, log_file=None):
    """Настраивает вывод журнала в консоль и в файл.

    Параметры:
        level: уровень журналирования (строка), по умолчанию берётся из ENV;
        log_file: путь к файлу журнала, по умолчанию берётся из ENV.
    Возвращает:
        настроенный объект logging.Logger.
    """
    level_name = (level or LOG_LEVEL).upper()
    raw_path = log_file or LOG_FILE
    log_path = Path(raw_path)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path
    handlers = [logging.StreamHandler()]
    if str(log_path) not in ("", "none", "None"):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("part2")
    logger.setLevel(level_name)
    if not logger.handlers:
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger
