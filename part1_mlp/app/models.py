"""Функции активации и класс многослойного перцептрона."""

import json

import numpy as np

from .exceptions import (
    FileOperationError,
    InvalidLayerSizeError,
    MismatchedDataError,
)


def sigmoid(value):
    """Логистическая функция активации."""
    return 1.0 / (1.0 + np.exp(-np.clip(value, -500.0, 500.0)))


def relu(value):
    """Выпрямленный линейный блок (ReLU)."""
    return np.maximum(0.0, value)


def sigmoid_derivative(value):
    """Производная сигмоиды по значению до активации."""
    activated = sigmoid(value)
    return activated * (1.0 - activated)


def relu_derivative(value):
    """Производная ReLU по значению до активации."""
    return (value > 0).astype(float)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
}


class NeuralNetwork:
    """Полносвязная нейронная сеть с произвольным числом слоёв.

    Параметры:
        layer_sizes: размеры слоёв, включая входной и выходной;
        activation: имя функции активации скрытых слоёв;
        random_seed: зерно генератора случайных чисел.
    """

    def __init__(self, layer_sizes, activation="sigmoid", random_seed=42):
        if len(layer_sizes) < 3:
            raise InvalidLayerSizeError(
                "Нужен как минимум один скрытый слой: например 2 4 1."
            )
        if any(int(size) <= 0 for size in layer_sizes):
            raise InvalidLayerSizeError("Размеры слоёв должны быть положительными.")
        if activation not in ACTIVATIONS:
            raise InvalidLayerSizeError(
                "Неизвестная активация: %s (доступно: %s)"
                % (activation, ", ".join(ACTIVATIONS))
            )
        self.layer_sizes = [int(size) for size in layer_sizes]
        self.activation_name = activation
        self.activation, self.activation_derivative = ACTIVATIONS[activation]
        self.random_seed = random_seed
        generator = np.random.default_rng(random_seed)
        self.weights = [
            generator.uniform(
                -1.0, 1.0, (self.layer_sizes[i], self.layer_sizes[i + 1])
            )
            for i in range(len(self.layer_sizes) - 1)
        ]
        self.biases = [np.zeros(size) for size in self.layer_sizes[1:]]
        self.loss_history = []
        self._activations = None
        self._pre_activations = None

    def _output_activation(self, value):
        """Активация выходного слоя (сигмоида — удобно для задач 0/1)."""
        return sigmoid(value)

    def _output_activation_derivative(self, value):
        """Производная активации выходного слоя."""
        return sigmoid_derivative(value)

    def forward(self, features):
        """Прямое распространение: возвращает выход сети.

        Параметры:
            features: массив формы (число примеров, число входов) или 1D-вектор.
        Возвращает:
            массив формы (число примеров, размер выходного слоя).
        """
        data = np.asarray(features, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                "Сеть ждёт %d входов, получено %d."
                % (self.layer_sizes[0], data.shape[1])
            )
        activations = data.copy()
        self._activations = [activations]
        self._pre_activations = []
        last_index = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre = activations @ weight + bias
            self._pre_activations.append(pre)
            if index == last_index:
                activations = self._output_activation(pre)
            else:
                activations = self.activation(pre)
            self._activations.append(activations)
        return activations

    def backward(self, features, targets, learning_rate):
        """Обратное распространение ошибки и шаг градиентного спуска.

        Параметры:
            features: входные примеры;
            targets: правильные ответы той же формы, что выход сети;
            learning_rate: скорость обучения.
        Возвращает:
            среднеквадратичную ошибку на этом наборе.
        """
        targets = np.asarray(targets, dtype=float)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        output = self.forward(features)
        if output.shape != targets.shape:
            raise MismatchedDataError(
                "Размерность ответов %s не совпадает с выходом сети %s."
                % (targets.shape, output.shape)
            )
        examples = output.shape[0]
        delta = (output - targets) * self._output_activation_derivative(
            self._pre_activations[-1]
        )
        for index in range(len(self.weights) - 1, -1, -1):
            previous = self._activations[index]
            gradient_w = previous.T @ delta / examples
            gradient_b = delta.mean(axis=0)
            if index > 0:
                delta = (delta @ self.weights[index].T) * self.activation_derivative(
                    self._pre_activations[index - 1]
                )
            self.weights[index] -= learning_rate * gradient_w
            self.biases[index] -= learning_rate * gradient_b
        return float(np.mean((output - targets) ** 2))

    def train(self, features, targets, epochs=100, batch_size=8, learning_rate=0.1):
        """Обучение на батчах с сохранением истории ошибок.

        Параметры:
            features, targets: обучающая выборка;
            epochs: число проходов по выборке;
            batch_size: размер батча;
            learning_rate: скорость обучения.
        Возвращает:
            список средней ошибки по эпохам.
        """
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if features.shape[0] != targets.shape[0]:
            raise MismatchedDataError(
                "Число примеров (%d) не совпадает с числом ответов (%d)."
                % (features.shape[0], targets.shape[0])
            )
        generator = np.random.default_rng(self.random_seed)
        total = features.shape[0]
        self.loss_history = []
        for _epoch in range(int(epochs)):
            order = generator.permutation(total)
            batch_losses = []
            for start in range(0, total, batch_size):
                indices = order[start : start + batch_size]
                loss = self.backward(
                    features[indices], targets[indices], learning_rate
                )
                batch_losses.append(loss)
            self.loss_history.append(float(np.mean(batch_losses)))
        return self.loss_history

    def predict(self, features):
        """Предсказание сети для одного или нескольких примеров."""
        return self.forward(features)

    def evaluate(self, features, targets, threshold=0.5):
        """Точность классификации на выборке (порог 0.5)."""
        targets = np.asarray(targets, dtype=float)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        output = self.forward(features)
        predictions = (output >= threshold).astype(float)
        return float(np.mean(predictions == targets))

    def save_weights(self, path):
        """Сохранение весов, смещений и архитектуры в JSON-файл."""
        payload = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "loss_history": self.loss_history,
        }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as error:
            raise FileOperationError("Не удалось сохранить веса: %s" % error)

    def load_weights(self, path):
        """Загрузка весов из JSON-файла с проверкой совместимости."""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as error:
            raise FileOperationError("Не удалось открыть файл весов: %s" % error)
        except json.JSONDecodeError as error:
            raise FileOperationError("Файл весов повреждён: %s" % error)
        if list(payload.get("layer_sizes", [])) != self.layer_sizes:
            raise MismatchedDataError(
                "Архитектура файла %s не совпадает с текущей сетью."
                % path
            )
        self.weights = [np.asarray(w, dtype=float) for w in payload["weights"]]
        self.biases = [np.asarray(b, dtype=float) for b in payload["biases"]]
        self.loss_history = payload.get("loss_history", [])
