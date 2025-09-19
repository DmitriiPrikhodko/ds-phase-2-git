import gradio as gr
import torch
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# === Настройка модели Detectron2 ===
cfg = get_cfg()
# Загружаем стандартный конфиг mask_rcnn_R_101_C4_3x
# from detectron2 import model_zoo
cfg.merge_from_file("day3_detectron/config_final.yaml")

# Меняем параметры под твою модель
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # <-- Укажи количество твоих классов
cfg.MODEL.WEIGHTS = "/home/dmitry/elbrus/Phase2/ds-phase-2-git/week2/day3_detectron/output/model_final.pth"  # <-- твои веса
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # порог уверенности
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# Настроим имена классов (чтобы красиво отображать)
dataset_name = cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) > 0 else "__unused"
MetadataCatalog.get(dataset_name).thing_classes = [
    "Heroes-3-objects-on-map",
    "Hero",
    "Mine",
    "Town",
]
metadata = MetadataCatalog.get(dataset_name)


# === Функция предсказания ===
def infer(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    outputs = predictor(image_bgr)

    v = Visualizer(
        image_bgr,
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    return cv2.cvtColor(v.get_image(), cv2.COLOR_BGR2RGB)


# === Интерфейс Gradio ===
demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Detectron2 Mask R-CNN | Heroes 3 Map",
    description="Загрузи изображение карты, и модель покажет найденные объекты.",
)

if __name__ == "__main__":
    demo.launch(share=True)
