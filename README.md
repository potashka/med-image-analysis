# Med Image Analysis Service

Этот проект представляет собой сервис для анализа медицинских изображений с использованием трех моделей: DeepHealth, DeepMED и NiftyNet.

## Требования

- Python 3.x
- Виртуальное окружение
- Установленные библиотеки из файла `requirements.txt`

## Установка и запуск

### Шаг 1: Клонирование репозитория

Клонируйте репозиторий на свой локальный компьютер:


git clone https://github.com/potashka/med-image-analysis.git
cd med-image-analysis

### Шаг 2: Создание и активация виртуального окружения
Создайте виртуальное окружение и активируйте его:


python -m venv myenv
myenv\Scripts\activate  # Для Windows
# source myenv/bin/activate  # Для Linux/Mac

### Шаг 3: Установка зависимостей
Установите необходимые библиотеки:

pip install -r requirements.txt


### Шаг 4: Запуск приложения
Запустите Flask приложение:

python app.py