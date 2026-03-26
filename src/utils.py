import yaml


def get_config(path):
    """Возвращает конфиги проекта."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        print("Конфигурационные данные успешно загружены.")
        return config
    except FileNotFoundError:
        print("Файл не найден.")
        return None
    except Exception as e:
        print("При открытии файла произошла ошибка.")
        return None
