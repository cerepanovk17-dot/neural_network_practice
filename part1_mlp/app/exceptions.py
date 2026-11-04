"""Пользовательские исключения приложения."""


class NeuralNetworkError(Exception):
    """Базовое исключение для всех ошибок приложения."""


class InvalidLayerSizeError(NeuralNetworkError):
    """Ошибка: некорректные размеры слоёв или имя активации."""


class MismatchedDataError(NeuralNetworkError):
    """Ошибка: размерности данных не совпадают с архитектурой сети."""


class DataLoadingError(NeuralNetworkError):
    """Ошибка: файл данных не найден или повреждён."""


class FileOperationError(NeuralNetworkError):
    """Ошибка: не удалось сохранить или загрузить файл весов."""


class InputValidationError(NeuralNetworkError):
    """Ошибка: некорректный ввод пользователя."""
