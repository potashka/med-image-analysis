"""
This module performs image segmentation using MONAI and NiftyNet models
and provides a web interface for predictions through a Flask application.
"""
import os
import sqlite3
from flask import Flask, request, jsonify, render_template
# import monai
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, ToTensor
from monai.networks.nets import UNet
import openslide
import torch
# from torch.utils.data import DataLoader
from niftynet.application.segmentation_application import SegmentationApplication  # type: ignore

os.add_dll_directory("C:\\Users\\avpot\\Downloads\\openslide-bin-4.0.0.2-windows-x64\\openslide-bin-4.0.0.2-windows-x64\\bin")
# os.add_dll_directory(r"C:\Users\avpot\Downloads\openslide-bin-4.0.0.2-windows-x64\openslide-bin-4.0.0.2-windows-x64\bin")


app = Flask(__name__)

# Определение трансформаций для MONAI
transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    ToTensor()
])

# Загрузка моделей
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
monai_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2)
).to(device)
monai_model.load_state_dict(torch.load("models/monai_model.pth"))
monai_model.eval()

# Инициализация приложения NiftyNet с параметрами
net_param = {
    'name': 'niftynet_unet',
    'activation_function': 'relu'
}
action_param = {
    'spatial_window_size': (64, 64, 64),
    'batch_size': 1
}
ACTION = 'inference'

niftynet_app = SegmentationApplication(
    net_param=net_param, action_param=action_param, action=ACTION
)
niftynet_app.initialise_application()

def init_db():
    """
    Создает таблицу results в базе данных, если она не существует.
    """
    if not os.path.exists('database'):
        os.makedirs('database')
    conn = sqlite3.connect("database/results.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            result TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_result(model, result):
    """
    Сохраняет результат предсказания в базу данных.

    Args:
        model (str): Название модели, использованной для предсказания.
        result (str): Результат предсказания.
    """
    conn = sqlite3.connect("database/results.db")
    c = conn.cursor()
    c.execute("INSERT INTO results (model, result) VALUES (?, ?)", (model, result))
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def index():
    """
    Отображает главную страницу веб-приложения.

    Returns:
        str: Содержимое HTML шаблона.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Обрабатывает запрос на предсказание, используя загруженное изображение.

    Returns:
        str: JSON-ответ с результатами предсказаний от моделей MONAI и NiftyNet.
    """
    image = request.files["image"]
    # Используем OpenSlide для открытия SVS-файла
    slide = openslide.OpenSlide(image)
    thumbnail = slide.get_thumbnail((1024, 1024))  # Извлекаем миниатюру
    # Преобразуем миниатюру в формат, совместимый с моделями
    img = transforms(thumbnail)
    img = img.unsqueeze(0).to(device)

    # Предсказание для модели MONAI
    with torch.no_grad():
        monai_prediction = monai_model.forward(img).cpu().numpy().tolist()
    save_result("MONAI", str(monai_prediction))

    # Обработка изображения для NiftyNet
    # niftynet_result = niftynet_app.run_inference(img)
    niftynet_result = niftynet_app.run_inference({'image': img.cpu().numpy()})
    niftynet_prediction = niftynet_result.tolist()
    save_result("NiftyNet", str(niftynet_prediction))

    return jsonify({
        "MONAI": monai_prediction,
        "NiftyNet": niftynet_prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
