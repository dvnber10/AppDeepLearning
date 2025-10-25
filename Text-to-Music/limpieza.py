from demucs import pretrained
from demucs.apply import apply_model
import torchaudio
import torch
from pathlib import Path

# 1. Cargar el modelo pre-entrenado
model = pretrained.load_model('htdemucs')

# 2. Especificar la ruta de tu archivo MP3 de entrada
archivo_entrada = "cancion.mp3"

# 3. Cargar el audio
wav, rate = torchaudio.load(archivo_entrada)

# Convertir a mono si es estéreo y asegurar formato correcto
if wav.dim() > 1 and wav.size(0) > 1:
    wav = wav.mean(dim=0, keepdim=True)

# 4. Aplicar el modelo para separar las pistas
with torch.no_grad():
    fuentes = apply_model(model, wav, device='cpu')  # Cambia a 'cuda' si tienes GPU

# 5. Guardar las pistas separadas
ruta_salida = Path("pistas_separadas")
ruta_salida.mkdir(exist_ok=True)

nombres = ["bateria.wav", "bajo.wav", "otros.wav", "voz.wav"]
for i, nombre in enumerate(nombres):
    pista_audio = fuentes[0, i]  # Extraer la pista específica
    torchaudio.save(str(ruta_salida / nombre), pista_audio.unsqueeze(0), rate)

print("¡Separación completada! Revisa la carpeta 'pistas_separadas'.")