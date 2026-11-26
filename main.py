import os

from google import genai
from google.genai import types

from products import products


STYLES = {
    "1": ("Рекламный", "Пиши привлекательно и дружелюбно, без навязчивых обещаний."),
    "2": ("Официальный", "Пиши нейтрально и точно, как редактор каталога."),
    "3": ("Юмористический", "Добавь лёгкий уместный юмор, сохраняя точность."),
}


def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        return

    with open(env_path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def get_text_models(client):
    models = []
    for model in client.models.list():
        name = getattr(model, "name", "").removeprefix("models/")
        methods = getattr(model, "supported_actions", []) or []
        if (
            name.startswith("gemini-")
            and "generateContent" in methods
            and not any(word in name for word in ("image", "banana", "tts", "live", "transcribe"))
        ):
            models.append(name)

    models = sorted(set(models))
    # Эта быстрая модель уже успешно отвечала с этим ключом.
    preferred = "gemini-3.1-flash-lite"
    if preferred in models:
        models.remove(preferred)
        models.insert(0, preferred)
    return models


def choose_model(models):
    if not models:
        raise SystemExit("Для API-ключа не найдено текстовых Gemini-моделей.")

    print("Доступные текстовые модели:")
    for number, model in enumerate(models, start=1):
        print(f"{number}. {model}")

    while True:
        value = input("Выбери модель [1]: ").strip() or "1"
        if value.isdigit() and 1 <= int(value) <= len(models):
            return models[int(value) - 1]
        print("Введи номер модели из списка.")


def choose_style():
    print("\nВыбери стиль описаний:")
    for number, (name, _) in STYLES.items():
        print(f"{number}. {name}")
    while True:
        choice = input("Твой выбор [1]: ").strip() or "1"
        if choice in STYLES:
            return STYLES[choice]
        print("Введи 1, 2 или 3.")


def choose_temperature():
    while True:
        value = input("Температура от 0 до 1 [0.7]: ").strip() or "0.7"
        try:
            temperature = float(value.replace(",", "."))
            if 0 <= temperature <= 1:
                return temperature
        except ValueError:
            pass
        print("Введи число от 0 до 1, например 0,7.")


def generate_description(client, models, selected_model, product, style_prompt, temperature):
    system_prompt = (
        "Ты копирайтер интернет-магазина. Составь описание товара ровно из двух предложений. "
        "Используй только сведения из названия товара: не придумывай материал, свойства, "
        "назначение, комплектность или другие характеристики. "
        f"Стиль: {style_prompt}"
    )
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=150,
    )

    candidates = [selected_model] + [model for model in models if model != selected_model]
    last_error = None
    for model in candidates:
        try:
            response = client.models.generate_content(
                model=model,
                contents=f"Название и характеристики товара: {product}",
                config=config,
            )
            return model, response
        except Exception as error:
            message = str(error)
            if not any(code in message for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")):
                raise
            last_error = error
            print(f"{model} временно недоступна, пробую следующую...")

    raise RuntimeError(f"Не удалось получить описание: {last_error}")


def main():
    load_env_file()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Добавь GEMINI_API_KEY в файл .env.")

    client = genai.Client(api_key=api_key)
    try:
        models = get_text_models(client)
    except Exception as error:
        raise SystemExit(f"Не удалось получить модели из Google AI: {error}")

    selected_model = choose_model(models)
    style_name, style_prompt = choose_style()
    temperature = choose_temperature()
    print(f"\nСоздаю описания: стиль «{style_name}», температура {temperature}.\n")

    total_tokens = 0
    results = [
        "Генератор описаний товаров",
        f"Модель: {selected_model}",
        f"Стиль: {style_name}",
        f"Температура: {temperature}",
        "",
    ]

    for product in products:
        print(f"Товар: {product}")
        try:
            selected_model, response = generate_description(
                client, models, selected_model, product, style_prompt, temperature
            )
            description = (response.text or "").strip()
            usage = getattr(response, "usage_metadata", None)
            total_tokens += getattr(usage, "total_token_count", 0) or 0
            print(f"Описание: {description}\n")
            results.extend([f"Товар: {product}", f"Описание: {description}", ""])
        except Exception as error:
            print(f"Не удалось создать описание: {error}\n")
            results.extend([f"Товар: {product}", f"Ошибка: {error}", ""])

    print(f"Всего токенов: {total_tokens}")
    results.append(f"Всего токенов: {total_tokens}")
    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w", encoding="utf-8") as results_file:
        results_file.write("\n".join(results) + "\n")
    print("Результаты сохранены в results.txt")


if __name__ == "__main__":
    main()
