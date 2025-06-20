import os
import logging
import tempfile
import uuid
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import BadRequest
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import nest_asyncio
import time
from torch.cuda.amp import autocast, GradScaler
nest_asyncio.apply()

# --- Настройки ---
TOKEN = "8025680437:AAEYu1to-5UQeKnUyV74nUeVYZyUFYJKuSA"
TEMP_DIR = "temp"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(TEMP_DIR, exist_ok=True)

STYLE_MODELS = {
    "candy": "models/candy_jit.pt",
    "mosaic": "models/mosaic_jit.pt",
    "udnie": "models/udnie_jit.pt",
    "lazy": "models/lazy_jit.pt",
    "rainprincess": "models/rain_princess_jit.pt",
    "starry": "models/starry_jit.pt",
    "tokyoghoul": "models/tokyo_ghoul_jit.pt",
    "wave": "models/wave_jit.pt"
}

logging.basicConfig(level=logging.INFO)

# --- Queue for Stylization Tasks ---
task_queue = asyncio.Queue()

# --- Модель FST ---
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

# --- NST Классы ---
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)
    def forward(self, img): return (img - self.mean) / self.std

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# --- Стилизация ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
import time
from torch.amp import autocast


class StyleTransferer:
    def __init__(self, image_width=1024, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Проверка доступной памяти и корректировка размера изображения
        if self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # В ГБ
            if total_memory < 4:  # Если памяти меньше 4 ГБ
                self.imsize = min(image_width, 512)  # Уменьшаем размер
            elif total_memory < 8:
                self.imsize = min(image_width, 768)
            else:
                self.imsize = image_width
        else:
            self.imsize = min(image_width, 512)  # Для CPU используем меньший размер
        self.loader = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.ToTensor()
        ])
        self.cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(self.device)
        self.cnn_normalization_mean = [0.485, 0.456, 0.406]
        self.cnn_normalization_std = [0.229, 0.224, 0.225]
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def image_loader(self, image):
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float32)  # Use FP32 to avoid FP16 issues

    def get_style_model_and_losses(self, style_img, content_img):
        if style_img.shape[1] != 3:
            style_img = style_img[:, :3, :, :]
        if content_img.shape[1] != 3:
            content_img = content_img[:, :3, :, :]

        model = nn.Sequential(
            Normalization(self.cnn_normalization_mean,
                          self.cnn_normalization_std,
                          self.device)
        ).to(self.device, torch.float32)  # Use FP32 for model

        content_losses = []
        style_losses = []

        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                continue

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]
        return model, style_losses, content_losses

    def run_style_transfer(self, content_img, style_img, num_steps=100, style_weight=1_000_000, content_weight=1):
        start_time = time.time()
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img)

        # Create a new leaf tensor
        input_img = content_img.detach().clone().to(device=self.device, dtype=torch.float32).requires_grad_(True)

        # Проверка использования памяти
        if self.device.type == "cuda":
            try:
                torch.cuda.memory_reserved()  # Проверка доступной памяти
            except RuntimeError:
                logging.warning("Недостаточно VRAM, переносим на CPU")
                self.device = torch.device("cpu")
                model = model.to(self.device)
                input_img = input_img.to(self.device)
                for sl in style_losses:
                    sl.target = sl.target.to(self.device)
                for cl in content_losses:
                    cl.target = cl.target.to(self.device)

        optimizer = optim.LBFGS([input_img])

        run = [0]
        while run[0] <= num_steps:
            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                loss = style_score * style_weight + content_score * content_weight
                if not isinstance(loss, torch.Tensor):
                    raise ValueError(f"Loss is not a tensor: {loss}")

                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    logging.info(
                        f"Step {run[0]}: Style Loss = {style_score.item():.4f}, Content Loss = {content_score.item():.4f}")

                return loss

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        logging.info(f"Общее время стилизации: {time.time() - start_time:.2f} секунд")
        return input_img

    def stylize(self, content_path, style_path):
        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')

        orig_width, orig_height = content_img.size
        ratio = orig_width / orig_height
        new_height = self.imsize
        new_width = int(self.imsize * ratio)

        content_img = content_img.resize((new_width, new_height))
        style_img = style_img.resize((new_width, new_height))

        content_tensor = self.image_loader(content_img)
        style_tensor = self.image_loader(style_img)

        output = self.run_style_transfer(content_tensor, style_tensor)

        output = output.squeeze(0).cpu().detach().to(torch.float32).permute(1, 2, 0).numpy()
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)
        return Image.fromarray(output)

    def stylize_fst(self, content_path, model_path):
        image = Image.open(content_path).convert('RGB')
        orig_width, orig_height = image.size
        transform = transforms.Compose([
            transforms.Resize(max(orig_width, orig_height)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        tensor = transform(image).unsqueeze(0).to(self.device)
        model = torch.jit.load(model_path, map_location=self.device).eval()
        with torch.no_grad():
            output = model(tensor).clamp(0, 255).cpu()

        output = output.squeeze().permute(1, 2, 0).numpy()
        output = np.clip(output, 0, 255).astype(np.uint8)
        result = Image.fromarray(output)
        return result.resize((orig_width, orig_height), Image.LANCZOS)

# --- Telegram Bot ---
class UserState:
    def __init__(self):
        self.content_image = None
        self.style_image = None
        self.mode = None
        self.selected_style = None
        self.previous_state = None
        self.last_message_id = None  # Track the last message ID

user_states = {}

def get_main_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📸 Начать стилизацию", callback_data="start_stylization")],
        [InlineKeyboardButton("ℹ️ О боте", callback_data="about")]
    ])

def get_back_button(previous_state):
    if not previous_state:
        return get_main_menu()
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⬅️ Назад", callback_data=f"back_{previous_state}")]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_states[user_id] = UserState()
    message = await update.message.reply_text(
        "👋 Добро пожаловать в бот стилизации изображений!\n"
        "Я могу преобразовать ваши фотографии в различные художественные стили.\n"
        "Выберите действие ниже:",
        reply_markup=get_main_menu()
    )
    user_states[user_id].last_message_id = message.message_id

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except BadRequest as e:
        logging.warning(f"Failed to answer callback query: {e}")
        return

    user_id = query.from_user.id
    state = user_states.setdefault(user_id, UserState())
    state.previous_state = "main_menu"
    try:
        await query.edit_message_text(
            "ℹ️ **О боте**\n\n"
            "Этот бот использует нейронные сети для стилизации изображений.\n"
            "Поддерживаемые функции:\n"
            "- **Fast Style Transfer (FST)**: Быстрая стилизация с предобученными стилями (Конфетный, Мозаика, Звёздная ночь и др.).\n"
            "- **Neural Style Transfer (NST)**: Стилизация с использованием вашего собственного изображения стиля (более длительный процесс).\n\n"
            "Как использовать:\n"
            "1. Нажмите 'Начать стилизацию' и отправьте изображение.\n"
            "2. Выберите стиль из предложенных или загрузите свой для NST.\n"
            "3. Дождитесь результата!\n\n"
            "📝 Бот создан для творчества и экспериментов с изображениями. Поддерживаются форматы JPEG и PNG.",
            reply_markup=get_back_button("main_menu")
        )
        state.last_message_id = query.message.message_id
    except BadRequest as e:
        logging.warning(f"Failed to edit message in about: {e}")
        await query.message.reply_text(
            "ℹ️ **О боте**\n\n"
            "Этот бот использует нейронные сети для стилизации изображений.\n"
            "Поддерживаемые функции:\n"
            "- **Fast Style Transfer (FST)**: Быстрая стилизация с предобученными стилями.\n"
            "- **Neural Style Transfer (NST)**: Стилизация с вашим изображением стиля.\n\n"
            "Как использовать:\n"
            "1. Нажмите 'Начать стилизацию' и отправьте изображение.\n"
            "2. Выберите стиль или загрузите свой для NST.\n"
            "3. Дождитесь результата!",
            reply_markup=get_back_button("main_menu")
        )
        state.last_message_id = query.message.message_id

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, UserState())

    photo = update.message.photo[-1] if update.message.photo else None
    doc = update.message.document

    if not photo and not (doc and doc.mime_type.startswith("image")):
        message = await update.message.reply_text(
            "❌ Пожалуйста, отправьте изображение (JPEG/PNG)",
            reply_markup=get_back_button(state.previous_state or "main_menu")
        )
        state.last_message_id = message.message_id
        return

    file = await context.bot.get_file(photo.file_id if photo else doc.file_id)
    ext = '.jpg'
    with tempfile.NamedTemporaryFile(dir=TEMP_DIR, suffix=ext, delete=False) as f:
        await file.download_to_memory(out=f)
        temp_path = f.name

    if not state.content_image:
        state.content_image = temp_path
        state.previous_state = "main_menu"
        await send_style_menu(update, context)
    elif not state.style_image and state.mode == "nst":
        state.style_image = temp_path
        await enqueue_task(update, context, user_id)

async def send_style_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, UserState())
    state.previous_state = "content_image"
    keyboard = [
        [InlineKeyboardButton("🍭 Конфетный", callback_data="fst_candy"),
         InlineKeyboardButton("🧩 Мозаика", callback_data="fst_mosaic")],
        [InlineKeyboardButton("🖌️ Импрессионизм", callback_data="fst_lazy"),
         InlineKeyboardButton("👑 Дождевая принцесса", callback_data="fst_rainprincess")],
        [InlineKeyboardButton("🌌 Звёздная ночь", callback_data="fst_starry"),
         InlineKeyboardButton("👹 Токийский гуль", callback_data="fst_tokyoghoul")],
        [InlineKeyboardButton("🌊 Волна", callback_data="fst_wave"),
         InlineKeyboardButton("🎭 Абстракция", callback_data="fst_udnie")],
        [InlineKeyboardButton("✨ Свой стиль (NST)", callback_data="nst")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_content_image")]
    ]
    try:
        if hasattr(update, 'message'):
            message = await update.message.reply_text(
                "Выберите стиль для стилизации изображения:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            message = await update.edit_message_text(
                "Выберите стиль для стилизации изображения:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        state.last_message_id = message.message_id
    except BadRequest as e:
        logging.warning(f"Failed to send/edit style menu: {e}")
        message = await update.message.reply_text(
            "Выберите стиль для стилизации изображения:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        state.last_message_id = message.message_id

async def enqueue_task(update_or_query, context: ContextTypes.DEFAULT_TYPE, user_id):
    state = user_states[user_id]
    try:
        message = update_or_query.message if hasattr(update_or_query, "message") else update_or_query
        reply = await message.reply_text(
            f"📥 Ваш запрос на стилизацию добавлен в очередь. Позиция: {task_queue.qsize() + 1}.",
            reply_markup=get_back_button(state.previous_state or "main_menu")
        )
        state.last_message_id = reply.message_id
        await task_queue.put({
            'user_id': user_id,
            'message': message,
            'content_image': state.content_image,
            'style_image': state.style_image,
            'mode': state.mode,
            'selected_style': state.selected_style,
            'previous_state': state.previous_state,
            'chat_id': message.chat_id,
            'message_id': state.last_message_id
        })
    except BadRequest as e:
        logging.error(f"Failed to enqueue task for user {user_id}: {e}")
        await message.reply_text(
            "❌ Не удалось добавить запрос в очередь. Попробуйте снова.",
            reply_markup=get_back_button(state.previous_state or "main_menu")
        )

async def process_task(task, context: ContextTypes.DEFAULT_TYPE):
    user_id = task['user_id']
    result_path = None
    try:
        processing_msg = await context.bot.send_message(
            chat_id=task['chat_id'],
            text="🛠 Начинается стилизация...",
            reply_to_message_id=task['message_id'],
            reply_markup=get_back_button(task['previous_state'] or "main_menu")
        )
        transferer = StyleTransferer()
        start_time = time.time()

        if task['mode'] == "fst":
            model_path = STYLE_MODELS[task['selected_style']]
            output = transferer.stylize_fst(task['content_image'], model_path)
        else:
            output = transferer.stylize(task['content_image'], task['style_image'])

        elapsed_time = time.time() - start_time
        result_path = os.path.join(TEMP_DIR, f"result_{user_id}_{uuid.uuid4().hex}.jpg")
        output.save(result_path)

        await context.bot.send_photo(
            chat_id=task['chat_id'],
            photo=result_path,
            caption=f"✨ Готово!\n⏱ Время стилизации: {elapsed_time:.2f} секунд",
            reply_to_message_id=task['message_id'],
            reply_markup=get_main_menu()
        )
        await processing_msg.delete()
    except Exception as e:
        logging.error(f"Ошибка стилизации для пользователя {user_id}: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=task['chat_id'],
            text="❌ Ошибка обработки.",
            reply_to_message_id=task['message_id'],
            reply_markup=get_back_button(task['previous_state'] or "main_menu")
        )
    finally:
        # Очистка памяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for path in [task['content_image'], task['style_image'], result_path]:
            if path and os.path.exists(path):
                os.remove(path)
        user_states.pop(user_id, None)

async def worker(context: ContextTypes.DEFAULT_TYPE):
    while True:
        task = await task_queue.get()
        try:
            await process_task(task, context)
        except Exception as e:
            logging.error(f"Ошибка обработки задачи в worker: {e}", exc_info=True)
        finally:
            task_queue.task_done()

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Update {update} caused error {context.error}")
    if update and (update.message or update.callback_query):
        try:
            chat_id = update.effective_chat.id
            if isinstance(context.error, BadRequest) and "query is too old" in str(context.error).lower():
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="⏳ Запрос устарел. Пожалуйста, начните заново.",
                    reply_markup=get_main_menu()
                )
            elif isinstance(context.error, BadRequest) and "no text in the message to edit" in str(context.error).lower():
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="❌ Не удалось обновить сообщение. Пожалуйста, начните заново.",
                    reply_markup=get_main_menu()
                )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="❌ Произошла ошибка. Попробуйте снова.",
                    reply_markup=get_main_menu()
                )
        except Exception as e:
            logging.error(f"Error in error_handler: {e}")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except BadRequest as e:
        logging.warning(f"Failed to answer callback query: {e}")
        return

    user_id = query.from_user.id
    state = user_states.setdefault(user_id, UserState())
    data = query.data

    logging.info(f"Button clicked by user {user_id}: {data}")

    try:
        if data == "start_stylization":
            state.style_image = None
            state.mode = None
            state.selected_style = None
            state.previous_state = "main_menu"
            try:
                await query.edit_message_text(
                    "📸 Отправьте изображение для стилизации.",
                    reply_markup=get_back_button("main_menu")
                )
                state.last_message_id = query.message.message_id
            except BadRequest as e:
                logging.warning(f"Failed to edit message: {e}")
                message = await query.message.reply_text(
                    "📸 Отправьте изображение для стилизации.",
                    reply_markup=get_back_button("main_menu")
                )
                state.last_message_id = message.message_id

        elif data == "about":
            await about(update, context)

        elif data.startswith("back_"):
            previous_state = data.split("_")[1]
            logging.info(f"Navigating back to {previous_state} for user {user_id}")
            if previous_state == "main_menu":
                if state.content_image and os.path.exists(state.content_image):
                    os.remove(state.content_image)
                if state.style_image and os.path.exists(state.style_image):
                    os.remove(state.style_image)
                state.content_image = None
                state.style_image = None
                state.mode = None
                state.selected_style = None
                state.previous_state = None
                try:
                    await query.edit_message_text(
                        "Вы вернулись в главное меню. Выберите действие:",
                        reply_markup=get_main_menu()
                    )
                    state.last_message_id = query.message.message_id
                except BadRequest as e:
                    logging.warning(f"Failed to edit message: {e}")
                    message = await query.message.reply_text(
                        "Вы вернулись в главное меню. Выберите действие:",
                        reply_markup=get_main_menu()
                    )
                    state.last_message_id = message.message_id

            elif previous_state == "content_image" and state.content_image:
                if state.style_image and os.path.exists(state.style_image):
                    os.remove(state.style_image)
                state.style_image = None
                state.mode = None
                state.selected_style = None
                state.previous_state = "main_menu"
                try:
                    await query.edit_message_text(
                        "📸 Отправьте новое изображение для стилизации или выберите стиль:",
                        reply_markup=get_back_button("main_menu")
                    )
                    state.last_message_id = query.message.message_id
                except BadRequest as e:
                    logging.warning(f"Failed to edit message: {e}")
                    message = await query.message.reply_text(
                        "📸 Отправьте новое изображение для стилизации или выберите стиль:",
                        reply_markup=get_back_button("main_menu")
                    )
                    state.last_message_id = message.message_id
                await send_style_menu(query, context)

            elif previous_state == "style_selection" and state.content_image:
                if state.style_image and os.path.exists(state.style_image):
                    os.remove(state.style_image)
                state.style_image = None
                state.mode = None
                state.selected_style = None
                try:
                    await query.edit_message_text(
                        "Выберите стиль для стилизации изображения:",
                        reply_markup=get_back_button("content_image")
                    )
                    state.last_message_id = query.message.message_id
                except BadRequest as e:
                    logging.warning(f"Failed to edit message: {e}")
                    message = await query.message.reply_text(
                        "Выберите стиль для стилизации изображения:",
                        reply_markup=get_back_button("content_image")
                    )
                    state.last_message_id = message.message_id
                await send_style_menu(query, context)

            else:
                try:
                    await query.edit_message_text(
                        "Вы вернулись в главное меню. Выберите действие:",
                        reply_markup=get_main_menu()
                    )
                    state.last_message_id = query.message.message_id
                except BadRequest as e:
                    logging.warning(f"Failed to edit message: {e}")
                    message = await query.message.reply_text(
                        "Вы вернулись в главное меню. Выберите действие:",
                        reply_markup=get_main_menu()
                    )
                    state.last_message_id = message.message_id

        elif data.startswith("fst_"):
            style = data.split("_")[1]
            state.mode = "fst"
            state.selected_style = style
            state.previous_state = "style_selection"
            try:
                await query.edit_message_text(
                    f"Выбран стиль: {style.capitalize()}",
                    reply_markup=get_back_button("content_image")
                )
                state.last_message_id = query.message.message_id
            except BadRequest as e:
                logging.warning(f"Failed to edit message: {e}")
                message = await query.message.reply_text(
                    f"Выбран стиль: {style.capitalize()}",
                    reply_markup=get_back_button("content_image")
                )
                state.last_message_id = message.message_id
            await enqueue_task(query, context, user_id)

        elif data == "nst":
            state.mode = "nst"
            state.previous_state = "style_selection"
            try:
                await query.edit_message_text(
                    "📷 Теперь загрузите изображение стиля.",
                    reply_markup=get_back_button("content_image")
                )
                state.last_message_id = query.message.message_id
            except BadRequest as e:
                logging.warning(f"Failed to edit message: {e}")
                message = await query.message.reply_text(
                    "📷 Теперь загрузите изображение стиля.",
                    reply_markup=get_back_button("content_image")
                )
                state.last_message_id = message.message_id

    except Exception as e:
        logging.error(f"Error in button_handler for user {user_id}: {e}", exc_info=True)
        await query.message.reply_text(
            "❌ Произошла ошибка. Попробуйте снова.",
            reply_markup=get_main_menu()
        )

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_error_handler(error_handler)

    # Используем post_init для запуска worker после инициализации приложения
    async def post_init(application):
        application.create_task(worker(application))

    app.post_init = post_init

    app.run_polling()

if __name__ == "__main__":
    main()