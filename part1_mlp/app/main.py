"""Консольное меню приложения части 1 (перцептрон).

Запуск из папки part1_mlp:
    python -m app.main
"""

import numpy as np

from .exceptions import InputValidationError, NeuralNetworkError
from .manager import DatasetManager
from .models import NeuralNetwork
from .utils import plot_loss, read_float, read_int


MENU = (
    "1. Создать сеть\n"
    "2. Загрузить данные из CSV\n"
    "3. Обучить сеть\n"
    "4. Сделать предсказание\n"
    "5. Сохранить веса / загрузить веса\n"
    "6. Показать график ошибки\n"
    "7. Выход"
)


def create_network():
    """Пункт меню 1: создание сети с пользовательскими параметрами."""
    raw = input("Размеры слоёв через пробел (например 2 4 1): ").strip()
    try:
        layer_sizes = [int(item) for item in raw.split()]
    except ValueError as error:
        raise InputValidationError("Размеры слоёв должны быть целыми числами.") from error
    activation = (input("Активация (sigmoid/relu) [sigmoid]: ").strip() or "sigmoid").lower()
    network = NeuralNetwork(layer_sizes, activation=activation)
    print(
        "Создана сеть: %s, активация: %s"
        % (" -> ".join(str(s) for s in network.layer_sizes), network.activation_name)
    )
    return network


def load_data(dataset):
    """Пункт меню 2: загрузка CSV с нормализацией."""
    path = input("Путь к CSV-файлу: ").strip()
    dataset.normalization = (
        input("Нормализация standard/minmax [standard]: ").strip() or "standard"
    ).lower()
    dataset.load_csv(path)
    dataset.normalize()
    print(
        "Загружено %d примеров, %d признаков, нормализация: %s"
        % (dataset.features.shape[0], dataset.features.shape[1], dataset.normalization)
    )


def train_network(network, dataset):
    """Пункт меню 3: обучение с выводом ошибки по эпохам."""
    epochs = read_int("Число эпох [200]: ", default=200)
    learning_rate = read_float("Скорость обучения [0.1]: ", default=0.1)
    batch_size = read_int("Размер батча [8]: ", default=8)
    X_train, X_test, y_train, y_test = dataset.split()
    history = network.train(X_train, y_train, epochs, batch_size, learning_rate)
    print(
        "Ошибка: первая эпоха %.6f, последняя %.6f, точность на тесте %.2f%%"
        % (history[0], history[-1], network.evaluate(X_test, y_test) * 100)
    )


def make_prediction(network, dataset):
    """Пункт меню 4: предсказание для примера, введённого вручную."""
    raw = input("Значения признаков через ; : ").strip()
    values = [float(item.replace(",", ".")) for item in raw.split(";")]
    if dataset._mean is not None:
        values = (np.asarray(values) - dataset._mean) / dataset._std
    output = network.predict(np.asarray(values, dtype=float))
    print("Выход сети: %s -> класс %d" % (np.round(output, 4), int(output >= 0.5)))


def save_or_load(network):
    """Пункт меню 5: сохранение или загрузка весов."""
    action = input("save / load: ").strip().lower()
    path = input("Путь к JSON-файлу весов [weights.json]: ").strip() or "weights.json"
    if action == "save":
        network.save_weights(path)
        print("Веса сохранены в %s" % path)
    elif action == "load":
        network.load_weights(path)
        print("Веса загружены из %s" % path)
    else:
        raise InputValidationError("Введите save или load.")


def show_plot(network):
    """Пункт меню 6: построение графика ошибки."""
    if not network.loss_history:
        print("Сначала обучите сеть.")
        return
    plot_loss(network.loss_history, "loss_history.png")
    print("График сохранён в loss_history.png")


def main():
    """Точка входа: цикл консольного меню."""
    network = None
    dataset = DatasetManager()
    while True:
        print("\n--- Меню ---")
        print(MENU)
        choice = input("> ").strip()
        try:
            if choice == "1":
                network = create_network()
            elif choice == "2":
                load_data(dataset)
            elif choice == "3":
                if network is None or dataset.features is None:
                    print("Сначала создайте сеть и загрузите данные (пункты 1 и 2).")
                else:
                    train_network(network, dataset)
            elif choice == "4":
                if network is None:
                    print("Сначала создайте сеть (пункт 1).")
                else:
                    make_prediction(network, dataset)
            elif choice == "5":
                if network is None:
                    print("Сначала создайте сеть (пункт 1).")
                else:
                    save_or_load(network)
            elif choice == "6":
                if network is None:
                    print("Сначала создайте сеть (пункт 1).")
                else:
                    show_plot(network)
            elif choice == "7":
                print("Выход.")
                return
            else:
                print("Нет такого пункта меню.")
        except (NeuralNetworkError, ValueError) as error:
            print("Ошибка: %s" % error)


if __name__ == "__main__":
    main()
