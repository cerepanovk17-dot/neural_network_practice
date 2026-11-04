# Практическое задание по дисциплине «Основы нейронных сетей»

Проект состоит из трёх самостоятельных частей, объединённых общими тестовыми данными
в папке `data/`.

## Структура проекта

```
neural_network_practice/
├── part1_mlp/          Часть 1: ООП-реализация многослойного перцептрона
│   └── app/            main.py, models.py, manager.py, exceptions.py, utils.py
├── part2_prepare/      Часть 2: CLI-утилита препроцессинга данных
│   └── app/            cli.py, config.py, scanner.py, logging_config.py
├── part3_parallel/     Часть 3: многопоточная и многопроцессорная аугментация
│   └── app/            producer_consumer.py, augment.py, performance.py
├── data/
│   ├── input/          тестовые данные (CSV и изображения PNG)
│   └── output/         результаты (npy, веса, отчёты, график ошибки)
├── logs/               файл журнала app.log
├── requirements.txt
├── .gitignore
└── .env.example
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # при необходимости настройте переменные
```

## Запуск

Часть 1 (консольное меню):

```bash
cd part1_mlp
python -m app.main
```

Часть 2 (CLI-утилита):

```bash
cd part2_prepare
python -m app.cli prepare --path ../data/input --ext png,csv --output ../data/output/dataset.npy
python -m app.cli doctor
```

Часть 3 (параллельная обработка):

```bash
cd part3_parallel
python -m app.producer_consumer --path ../data/input --workers 2
python -m app.performance --path ../data/input --output ../data/output/performance_report.json
```

## Примечания

- Все пути и параметры настраиваются через аргументы командной строки и переменные
  окружения (см. `.env.example`); в коде нет захардкоженных путей.
- Если пул процессов недоступен в среде исполнения (например, в ограниченном контейнере
  без системных семафоров), программа части 3 автоматически переключается на пул потоков
  с предупреждением в консоли. На обычной рабочей машине используется пул процессов.
- Файл `data/output/performance_report.json`, приложенный к проекту, получен в тестовой
  среде; перед сдачей запустите `app.performance` на своей машине и обновите отчёт.
