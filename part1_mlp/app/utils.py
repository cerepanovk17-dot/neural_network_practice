"""Вспомогательные функции: график ошибки и безопасный ввод."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .exceptions import InputValidationError  # noqa: E402


def plot_loss(history, path="loss_history.png"):
    """Сохранение графика ошибки обучения в PNG-файл.

    Параметры:
        history: список значений ошибки по эпохам;
        path: имя файла для сохранения.
    """
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history, color="tab:blue")
    axis.set_xlabel("Эпоха")
    axis.set_ylabel("Среднеквадратичная ошибка")
    axis.set_title("История ошибки обучения")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def read_float(prompt, default=None):
    """Чтение дробного числа с проверкой корректности ввода."""
    raw = input(prompt).strip()
    if not raw and default is not None:
        return float(default)
    try:
        return float(raw.replace(",", "."))
    except ValueError as error:
        raise InputValidationError("Ожидалось число, получено: %r" % raw) from error


def read_int(prompt, default=None):
    """Чтение целого числа с проверкой корректности ввода."""
    raw = input(prompt).strip()
    if not raw and default is not None:
        return int(default)
    try:
        return int(raw)
    except ValueError as error:
        raise InputValidationError("Ожидалось целое число, получено: %r" % raw) from error
