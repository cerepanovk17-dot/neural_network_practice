"""Загрузка, нормализация и разбиение табличных данных."""

import csv
import os

import numpy as np

from .exceptions import DataLoadingError


class DatasetManager:
    """Менеджер данных: CSV-файл, нормализация признаков, train/test.

    Параметры:
        normalization: способ нормализации признаков — 'standard' или 'minmax'.
    """

    def __init__(self, normalization="standard"):
        if normalization not in ("standard", "minmax"):
            raise DataLoadingError(
                "Способ нормализации может быть 'standard' или 'minmax'."
            )
        self.normalization = normalization
        self.feature_names = []
        self.features = None
        self.targets = None
        self._mean = None
        self._std = None
        self._min = None
        self._max = None

    def load_csv(self, path):
        """Чтение CSV: первая строка — заголовки, последний столбец — ответ.

        Параметры:
            path: путь к CSV-файлу.
        Возвращает:
            кортеж (признаки, ответы).
        """
        if not os.path.isfile(path):
            raise DataLoadingError("Файл не найден: %s" % path)
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter=";")
                header = next(reader)
                rows = [row for row in reader if any(cell.strip() for cell in row)]
        except (OSError, StopIteration) as error:
            raise DataLoadingError("Не удалось прочитать CSV: %s" % error)
        if len(header) < 2:
            raise DataLoadingError("В файле должен быть минимум один признак и ответ.")
        self.feature_names = [name.strip() for name in header]
        parsed = []
        for number, row in enumerate(rows, start=2):
            if len(row) != len(header):
                raise DataLoadingError(
                    "Строка %d: столбцов %d вместо %d." % (number, len(row), len(header))
                )
            try:
                parsed.append([float(cell.replace(",", ".")) for cell in row])
            except ValueError as error:
                raise DataLoadingError("Строка %d: не число (%s)." % (number, error))
        table = np.asarray(parsed, dtype=float)
        self.features = table[:, :-1]
        self.targets = table[:, -1:]
        return self.features, self.targets

    def normalize(self):
        """Нормализация признаков по параметрам, посчитанным на данных.

        Возвращает:
            нормализованные признаки.
        """
        if self.features is None:
            raise DataLoadingError("Сначала загрузите данные из CSV.")
        if self.normalization == "standard":
            self._mean = self.features.mean(axis=0)
            self._std = self.features.std(axis=0)
            self._std[self._std == 0] = 1.0
            self.features = (self.features - self._mean) / self._std
        else:
            self._min = self.features.min(axis=0)
            self._max = self.features.max(axis=0)
            span = self._max - self._min
            span[span == 0] = 1.0
            self.features = (self.features - self._min) / span
        return self.features

    def split(self, train_ratio=0.8, seed=42):
        """Деление выборки на обучающую и контрольную части.

        Параметры:
            train_ratio: доля обучающей выборки (0 < train_ratio < 1);
            seed: зерно перемешивания.
        Возвращает:
            кортеж (X_train, X_test, y_train, y_test).
        """
        if self.features is None:
            raise DataLoadingError("Сначала загрузите данные из CSV.")
        if not 0.0 < train_ratio < 1.0:
            raise DataLoadingError("Доля обучающей выборки должна быть между 0 и 1.")
        generator = np.random.default_rng(seed)
        total = self.features.shape[0]
        order = generator.permutation(total)
        border = int(total * train_ratio)
        train_indices = order[:border]
        test_indices = order[border:]
        return (
            self.features[train_indices],
            self.features[test_indices],
            self.targets[train_indices],
            self.targets[test_indices],
        )
