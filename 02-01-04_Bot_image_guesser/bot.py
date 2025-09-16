import logging
import os
import io
import asyncio
import multiprocessing as mp

import torch
from torch import nn
from torchvision import transforms as T
from torchvision.models import convnext_tiny
from PIL import Image

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import Message
from aiogram.filters import Command

# --- Настройка multiprocessing ---
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

# --- Настройка PyTorch ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)

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
TOKEN = "8369392797:AAHwTSC8JnwL8INDF-hOnuyzzKC3RqU_2wI"
WEIGHTS_PATH = "./weights/weights.pth"

CLASSES = {
    0: "алтарь",
    1: "апсида",
    2: "колокольня",
    3: "колонна",
    4: "купол (внутри)",
    5: "купол (снаружи)",
    6: "аркбутан",
    7: "гаргулья",
    8: "витраж",
    9: "свод",
}

# --- Глобальные переменные ---
model = None
transform = None


def load_model(weights_path: str):
    """Загружает модель один раз при старте"""
    logging.info("Загружаю модель...")
    m = convnext_tiny()
    m.to(DEVICE)
    m.classifier[2] = nn.Linear(768, 10)
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    m.eval()
    logging.info("Модель успешно загружена")
    return m


def create_transform():
    """Создаёт пайплайн преобразований"""
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_prediction(img: Image.Image):
    """Делает предсказание по изображению"""
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
    return CLASSES.get(pred_idx, "неизвестно"), confidence


async def download_photo(bot: Bot, file_id: str) -> Image.Image:
    """Скачивает фото и возвращает PIL Image"""
    file = await bot.get_file(file_id)
    photo_bytes = await bot.download_file(file.file_path)
    return Image.open(photo_bytes)


async def process_single_photo(bot: Bot, photo: types.PhotoSize, index: int):
    """Обрабатывает одно фото"""
    try:
        image = await download_photo(bot, photo.file_id)
        pred_class, confidence = get_prediction(image)
        return {
            "index": index,
            "pred_class": pred_class,
            "confidence": confidence,
            "success": True,
        }
    except Exception as e:
        return {"index": index, "error": str(e), "success": False}


# --- Хэндлеры ---


async def cmd_start(message: Message):
    user_name = message.from_user.full_name
    logging.info(f"{user_name} ({message.from_user.id}) запустил бота")
    await message.answer(
        f"Здравствуйте, {user_name}!\n"
        "Загрузите картинку, и я скажу, что на ней изображено.\n"
        "Можно отправлять несколько фото сразу!"
    )


async def text_answer(message: Message):
    user_name = message.from_user.full_name
    logging.info(f"{user_name} ({message.from_user.id}) написал: {message.text}")
    await message.answer("Я не умею разговаривать, умею только распознавать картинки 🖼️")


async def handle_photo(message: types.Message):
    """Обработка фото — берём только оригинальные версии (последний элемент message.photo)"""
    user_name = message.from_user.full_name
    logging.info(f"{user_name} загрузил фото ({len(message.photo)} размеров)")

    processing_msg = await message.answer("⏳ Обрабатываю фото...")

    # Берём только оригинальную версию (последний элемент)
    photo = message.photo[-1]
    result = await process_single_photo(message.bot, photo, 0)

    if result["success"]:
        response = (
            f"📸 Фото:\n"
            f"🏷️ Класс: {result['pred_class']}\n"
            f"✅ Уверенность: {result['confidence']:.2%}"
        )
    else:
        response = f"❌ Ошибка обработки: {result['error']}"

    await processing_msg.delete()
    await message.answer(response)


async def handle_album(messages: list[Message]):
    """Обработка альбома (нескольких фото за раз)"""
    bot = messages[0].bot
    user_name = messages[0].from_user.full_name
    logging.info(f"{user_name} загрузил альбом из {len(messages)} фото")

    # Отправляем уведомление о начале обработки
    processing_msg = await messages[0].answer(f"⏳ Обрабатываю {len(messages)} фото...")

    # Берём по одному оригинальному фото из каждого сообщения
    tasks = [process_single_photo(bot, m.photo[-1], i) for i, m in enumerate(messages)]
    results = await asyncio.gather(*tasks)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    response = f"📊 Результаты обработки {len(messages)} фото:\n\n"
    for r in successful:
        response += (
            f"📸 Фото {r['index']+1}:\n"
            f"   🏷️ Класс: {r['pred_class']}\n"
            f"   ✅ Уверенность: {r['confidence']:.2%}\n"
            f"   ────────────────\n"
        )
    if failed:
        response += f"\n❌ Не удалось обработать {len(failed)} фото\n"

    response += f"\n📈 Итого: {len(successful)} успешно, {len(failed)} с ошибками"

    await processing_msg.delete()
    await messages[0].answer(response)


async def handle_document(message: types.Message):
    """Обработка изображений, отправленных как файлы"""
    if message.document.mime_type and message.document.mime_type.startswith("image/"):
        try:
            image = await download_photo(message.bot, message.document.file_id)
            pred_class, confidence = get_prediction(image)
            await message.answer(
                f"🎯 Результат распознавания (документ):\n"
                f"📊 Класс: {pred_class}\n"
                f"✅ Уверенность: {confidence:.2%}"
            )
        except Exception as e:
            logging.error(f"Ошибка при обработке документа: {e}")
            await message.answer("❌ Произошла ошибка при обработке файла.")


async def main():
    global model, transform
    model = load_model(WEIGHTS_PATH)
    transform = create_transform()

    bot = Bot(token=TOKEN)
    dp = Dispatcher()

    dp.message.register(cmd_start, Command("start"))
    dp.message.register(text_answer, F.text)
    dp.message.register(handle_photo, F.photo)
    dp.message.register(handle_document, F.document)

    logging.info("Бот запущен 🚀")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
