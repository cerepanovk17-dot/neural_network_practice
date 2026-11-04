"""Аугментация изображений: повороты, отражения и добавление шума."""

import numpy as np


def augment_image(array):
    """Строит набор аугментированных копий одного изображения.

    Параметры:
        array: массив изображения произвольной формы со значениями в диапазоне 0..1.
    Возвращает:
        список из шести массивов: исходный, два поворота, два отражения и зашумлённый.
    """
    original = np.asarray(array, dtype=float)
    noise = np.random.default_rng().normal(0.0, 0.05, original.shape)
    return [
        original,
        np.rot90(original, 1).copy(),
        np.rot90(original, 2).copy(),
        np.fliplr(original).copy(),
        np.flipud(original).copy(),
        np.clip(original + noise, 0.0, 1.0),
    ]


def augment_one(args):
    """Обёртка для multiprocessing.Pool: принимает кортеж (индекс, массив).

    Возвращает:
        кортеж (индекс, список аугментированных массивов).
    """
    index, array = args
    return index, augment_image(array)
