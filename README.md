# Как запустить код?
Клонируем репозиторий себе на компьютер и переходим в папку с проектом
```bash
git clone https://github.com/eura17/ff-case.git 
cd ff-case
```
Устанавливаем необходимые зависимости и создаем ядро для Jupyter
```bash
poetry install
poetry run python -m ipykernel install --user --name "ff-case" --display-name "Python (ff-case)"
```

Запускаем Jupyter Lab/Notebook и открываем ноутбуки. 
В правом верхнем углу выбираем кернел, созданный ранее.
Все должно заработать.

P.s.: Предварительно нужно установить poetry, что делается одной командой с
[официального сайта](https://python-poetry.org/docs/). 

Также, очевидно, в папку с проектом надо закинуть папку `data`,
которая содержит подпапки `task1` и `task2` и файл `rf.csv` с годовой 
доходностью 10-летних бондов (я скачал с Investing.com и в коде привел значения
к дневной доходности). В папке `task1` точно такая же струкутура, как и в 
исходном архиве `Задание 1. Таймсерии.rar`, в  папке `task2` точно такая же 
структура как в исходном архиве `Задание 2. Финансовые данные.rar`.
