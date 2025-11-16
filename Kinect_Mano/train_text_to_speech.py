# train_tts.py
import matplotlib.pyplot as plt
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2

# Configuración
config = Tacotron2Config(
    audio=BaseAudioConfig(),
    run_name="tts_spanish",
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    eval_split_size=0.1,
    test_delay_epochs=-1,
    epochs=10,
    print_step=25,
    save_step=500,
    save_checkpoints=True,
    output_path="./tts_model",
)

# Modelo
model = Tacotron2(config)

# Dataset CSS10 español (descargar previamente y preparar en formato TTS)
config.datasets = [
    {
        "formatter": "ljspeech",
        "path": "./datasets/css10_spanish",
        "meta_file_train": "metadata.csv",
    }
]

# Entrenador
trainer = Trainer(TrainerArgs(), config, model)
trainer.fit()

# Graficar curvas de entrenamiento
train_loss = trainer.train_losses
eval_loss = trainer.eval_losses

plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Eval Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("TTS Training Loss")
plt.legend()
plt.savefig("tts_loss.png")

print("✅ TTS entrenado. Último error de validación:", eval_loss[-1])
