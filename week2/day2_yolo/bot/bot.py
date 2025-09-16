import logging
import os
import io
import asyncio
import torch
from PIL import Image

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Message
from aiogram.filters import Command


# --- Логирование ---
os.makedirs("./log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename="./log/bot_log.log",
    format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# --- Константы ---
TOKEN = "YOUR_BOT_TOKEN_HERE"
YOLO_WEIGHTS = "./weights/yolov8n_weights.pth"  # заглушка для весов

# --- Глобальные переменные ---
model = None

# --- Загрузка модели YOLO ---
def load_model(weights_path: str):
    """Загружает YOLO модель"""
    logging.info("Загружаю YOLO модель...")
    try:
        from ultralytics import YOLO
        m = YOLO(weights_path)  # заглушка, если весов нет, можно заменить на 'yolov8n.pt'
        logging.info("YOLO модель успешно загружена")
        return m
    except ImportError:
        logging.error("Ultralytics YOLO не установлен. Установите через pip install ultralytics")
        raise


async def download_photo(bot: Bot, file_id: str) -> Image.Image:
    """Скачивает фото и возвращает PIL Image"""
    file = await bot.get_file(file_id)
    photo_bytes = await bot.download_file(file.file_path)
    return Image.open(photo_bytes).convert("RGB")


async def process_single_photo(bot: Bot, photo: types.PhotoSize, index: int):
    """Обрабатывает одно фото и возвращает картинку с результатом"""
    try:
        image = await download_photo(bot, photo.file_id)
        # --- YOLO предсказание ---
        results = model.predict(image, imgsz=640, conf=0.25)  # параметры можно настроить
        # --- Рендерим результат на картинку ---
        annotated_frame = results[0].plot()  # numpy array
        annotated_image = Image.fromarray(annotated_frame)

        # Сохраняем в байтовый поток для отправки
        bio = io.BytesIO()
        annotated_image.save(bio, format="PNG")
        bio.seek(0)

        return {
            "index": index,
            "image_bytes": bio,
            "success": True,
        }
    except Exception as e:
        return {"index": index, "error": str(e), "success": False}


async def handle_photo(message: types.Message):
    """Обработка фото — берём только оригинальные версии (последний элемент message.photo)"""
    user_name = message.from_user.full_name
    logging.info(f"{user_name} загрузил фото ({len(message.photo)} размеров)")

    processing_msg = await message.answer("⏳ Обрабатываю фото...")

    # Берём только оригинальную версию (последний элемент)
    photo = message.photo[-1]
    result = await process_single_photo(message.bot, photo, 0)

    await processing_msg.delete()

    if result["success"]:
        await message.answer_photo(result["image_bytes"], caption="📸 Результат распознавания")
    else:
        await message.answer(f"❌ Ошибка обработки: {result['error']}")


async def cmd_start(message: Message):
    user_name = message.from_user.full_name
    logging.info(f"{user_name} ({message.from_user.id}) запустил бота")
    await message.answer(
        f"Здравствуйте, {user_name}!\n"
        "Загрузите картинку, и я покажу, что на ней обнаружил YOLO.\n"
        "Можно отправлять несколько фото сразу!"
    )


async def text_answer(message: Message):
    user_name = message.from_user.full_name
    logging.info(f"{user_name} ({message.from_user.id}) написал: {message.text}")
    await message.answer("Я не умею разговаривать, умею только распознавать объекты на картинках 🖼️")


async def main():
    global model
    model = load_model(YOLO_WEIGHTS)

    bot = Bot(token=TOKEN)
    dp = Dispatcher()

    dp.message.register(cmd_start, Command("start"))
    dp.message.register(text_answer, F.text)
    dp.message.register(handle_photo, F.photo)

    logging.info("Бот запущен 🚀")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
