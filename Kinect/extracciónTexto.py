from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# 🔹 Cargar modelo y procesador SOLO una vez (no en cada llamada)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def extract_text_from_image_1(image_path):
    """
    Extrae texto manuscrito desde una imagen usando TrOCR.
    """
    try:
        # Abrir imagen
        image = Image.open(image_path).convert("RGB")

        # Preparar imagen para el modelo
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        # Generar predicción
        generated_ids = model.generate(pixel_values)

        # Decodificar el texto
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text.strip()

    except Exception as e:
        print(f"[ERROR OCR] {e}")
        return ""
