import mimetypes
import os
from pathlib import Path

from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent
IMAGE_RESULT = BASE_DIR / "image_result.txt"
SPEECH_RESULT = BASE_DIR / "speech_result.txt"
RESULTS = BASE_DIR / "results.txt"

IMAGE_PROMPT = (
    "Подробно опиши фотографию на русском языке. Определи, что изображено, "
    "перечисли видимые предметы, укажи их количество, если это можно определить, "
    "и опиши расположение предметов. Не придумывай объекты и детали, которых "
    "нельзя уверенно увидеть на изображении."
)

OCR_PROMPT = (
    "Перепиши весь текст с изображения на русском языке, сохраняя порядок строк. "
    "Не придумывай отсутствующий текст. Если символ неразборчив, напиши "
    "[неразборчиво]. Если на изображении чек, после переписанного текста отдельно "
    "укажи название магазина, дату покупки, список товаров и итоговую сумму."
)

SPEECH_PROMPT = (
    "Дословно распознай речь в аудиозаписи и выведи только её текст на русском языке. "
    "Сохрани числа и имена так, как они произнесены. Не добавляй комментарии."
)


def load_env_file():
    env_path = BASE_DIR / ".env"
    if not env_path.is_file():
        return

    with env_path.open(encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def get_models(client):
    models = []
    for model in client.models.list():
        name = getattr(model, "name", "").removeprefix("models/")
        actions = getattr(model, "supported_actions", []) or []
        if (
            name.startswith("gemini-")
            and "generateContent" in actions
            and not any(
                word in name
                for word in ("image", "banana", "tts", "live", "transcribe")
            )
        ):
            models.append(name)

    models = sorted(set(models))
    # Используем уже проверенную на этом ключе быструю модель первой.
    preferred = "gemini-3.1-flash-lite"
    if preferred in models:
        models.remove(preferred)
        models.insert(0, preferred)
    return models


def ask_file(prompt, default_name):
    value = input(f"{prompt} [{default_name}]: ").strip()
    path = Path(value).expanduser() if value else BASE_DIR / default_name
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return path


def get_mime_type(path, media_type):
    if media_type == "audio":
        audio_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".webm": "audio/webm",
        }
        mime_type = audio_types.get(path.suffix.lower())
    else:
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type and not mime_type.startswith("image/"):
            mime_type = None

    if not mime_type:
        raise ValueError(f"Не удалось определить формат файла: {path.name}")
    return mime_type


def generate(client, models, selected_model, prompt, file_path, media_type):
    if media_type == "audio":
        # Files API упрощает передачу аудиозаписей разных поддерживаемых форматов.
        media_part = client.files.upload(file=file_path)
    else:
        mime_type = get_mime_type(file_path, media_type)
        with file_path.open("rb") as media_file:
            media_part = types.Part.from_bytes(data=media_file.read(), mime_type=mime_type)

    for model in [selected_model] + [item for item in models if item != selected_model]:
        try:
            contents = (
                [media_part, prompt]
                if media_type == "audio"
                else [prompt, media_part]
            )
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            description = (response.text or "").strip()
            if not description:
                raise RuntimeError("Gemini вернул пустой ответ.")
            return model, description
        except Exception as error:
            message = str(error)
            can_retry = any(
                marker in message
                for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")
            )
            if not can_retry:
                raise
            print(f"{model} временно недоступна, пробую следующую...")

    raise RuntimeError("Все доступные модели временно недоступны.")


def save_image_result(mode, file_path, model, answer):
    content = (
        f"Режим: {mode}\nФайл: {file_path.name}\nМодель: {model}\n\n"
        f"{answer}\n"
    )
    IMAGE_RESULT.write_text(content, encoding="utf-8")
    with RESULTS.open("a", encoding="utf-8") as results_file:
        results_file.write("\n" + "=" * 50 + "\n" + content)


def save_speech_result(file_path, model, transcript):
    content = f"Файл: {file_path.name}\nМодель: {model}\n\n{transcript}\n"
    SPEECH_RESULT.write_text(content, encoding="utf-8")


def process_image(client, models, selected_model, prompt, mode, default_file):
    file_path = ask_file("Путь к изображению", default_file)
    selected_model, answer = generate(
        client, models, selected_model, prompt, file_path, "image"
    )
    print(f"\nОтвет Gemini ({selected_model}):\n{answer}\n")
    save_image_result(mode, file_path, selected_model, answer)
    print("Результат сохранён в image_result.txt и добавлен в results.txt.\n")
    return selected_model


def process_speech(client, models, selected_model):
    file_path = ask_file("Путь к аудиозаписи", "voice.mp3")
    selected_model, transcript = generate(
        client, models, selected_model, SPEECH_PROMPT, file_path, "audio"
    )
    print(f"\nРаспознанный текст ({selected_model}):\n{transcript}\n")
    save_speech_result(file_path, selected_model, transcript)
    print("Текст сохранён в speech_result.txt.\n")
    return selected_model


def main():
    load_env_file()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Добавь GEMINI_API_KEY в файл .env.")

    client = genai.Client(api_key=api_key)
    try:
        models = get_models(client)
    except Exception as error:
        raise SystemExit(f"Не удалось получить список моделей Google AI: {error}")

    if not models:
        raise SystemExit("Google AI не вернул доступных Gemini-моделей.")
    selected_model = models[0]
    print(f"Модель по умолчанию: {selected_model}\n")
    while True:
        print("=== AI MEDIA ASSISTANT ===")
        print("1. Описать фотографию")
        print("2. Распознать текст на фотографии")
        print("3. Распознать аудиозапись")
        print("4. Выход")
        choice = input("\nВыберите действие: ").strip()

        try:
            if choice == "1":
                selected_model = process_image(
                    client, models, selected_model, IMAGE_PROMPT,
                    "Описание фотографии", "complex.jpg",
                )
            elif choice == "2":
                selected_model = process_image(
                    client, models, selected_model, OCR_PROMPT,
                    "Распознавание текста (OCR)", "check.jpg",
                )
            elif choice == "3":
                selected_model = process_speech(client, models, selected_model)
            elif choice == "4":
                print("Пока!")
                break
            else:
                print("Выбери пункт 1, 2, 3 или 4.\n")
        except Exception as error:
            print(f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
