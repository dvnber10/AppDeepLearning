import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from io import BytesIO
import asyncio
import platform

# Hiperparámetros
VOCAB_SIZE = 256  # Para caracteres ASCII
EMBED_DIM = 128
HIDDEN_DIM = 256
MEL_BINS = 80  # Para Mel-spectrogram
SAMPLE_RATE = 22050
PITCH_RANGE = 128  # MIDI pitches (0-127)

class TextEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

    def forward(self, text):
        return self.embed(text)  # [batch, seq_len, embed_dim]

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)

    def forward(self, x):
        output, _ = self.lstm(x)  # output: [batch, seq_len, 2*HIDDEN_DIM]
        return output

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(2 * HIDDEN_DIM + HIDDEN_DIM, HIDDEN_DIM)
        self.v = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, HIDDEN_DIM]
        # encoder_outputs: [batch, seq_len, 2*HIDDEN_DIM]
        query = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)  # [batch, seq_len, HIDDEN_DIM]
        combined = torch.cat((query, encoder_outputs), dim=2)  # [batch, seq_len, HIDDEN_DIM + 2*HIDDEN_DIM]
        energy = self.v(torch.tanh(self.W(combined))).squeeze(2)  # [batch, seq_len]
        attn_weights = F.softmax(energy, dim=1)  # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch, 2*HIDDEN_DIM]
        return context

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pitch_embed = nn.Embedding(PITCH_RANGE, EMBED_DIM)
        self.lstm = nn.LSTM(EMBED_DIM + 2 * HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, MEL_BINS)

    def forward(self, context, pitch, hidden):
        # context: [batch, 2*HIDDEN_DIM]
        # pitch: [batch]
        pitch_emb = self.pitch_embed(pitch)  # [batch, EMBED_DIM]
        input = torch.cat((pitch_emb, context), dim=1).unsqueeze(1)  # [batch, 1, EMBED_DIM + 2*HIDDEN_DIM]
        output, hidden = self.lstm(input, hidden)  # output: [batch, 1, HIDDEN_DIM]
        mel = self.fc(output.squeeze(1))  # [batch, MEL_BINS]
        return mel, hidden

class SingingTTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embed = TextEmbedding()
        self.encoder = Encoder()
        self.attention = Attention()
        self.decoder = Decoder()

    def forward(self, text, pitches):
        # text: [batch, seq_len]
        # pitches: [batch, seq_len]
        embedded = self.text_embed(text)  # [batch, seq_len, EMBED_DIM]
        enc_outputs = self.encoder(embedded)  # [batch, seq_len, 2*HIDDEN_DIM]
        batch_size = text.size(0)
        hidden = (torch.zeros(1, batch_size, HIDDEN_DIM), torch.zeros(1, batch_size, HIDDEN_DIM))
        mels = []
        for i in range(pitches.size(1)):
            context = self.attention(hidden[0].squeeze(0), enc_outputs)  # [batch, 2*HIDDEN_DIM]
            mel, hidden = self.decoder(context, pitches[:, i], hidden)  # mel: [batch, MEL_BINS]
            mels.append(mel)
        return torch.stack(mels, dim=1)  # [batch, seq_len, MEL_BINS]

class MusicGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, seq_len):
        input = torch.randn(1, seq_len, 1)  # Noise inicial
        output, _ = self.lstm(input)
        freqs = torch.sigmoid(self.fc(output)) * 1000 + 200  # Frecuencias entre 200-1200 Hz
        return freqs.squeeze()

def generate_waveform(freqs, duration_ms, sr=SAMPLE_RATE):
    t = np.linspace(0, duration_ms / 1000, int(sr * duration_ms / 1000), False)
    audio = np.zeros_like(t)
    segment_len = len(t) // len(freqs)
    for i, f in enumerate(freqs):
        segment = np.sin(2 * np.pi * f * t[i * segment_len:(i + 1) * segment_len])
        audio[i * segment_len:(i + 1) * segment_len] = segment
    return audio / np.max(np.abs(audio))

def griffin_lim(mel, n_iters=30):
    mel = mel.detach().numpy().T if torch.is_tensor(mel) else mel.T
    stft = librosa.feature.inverse.mel_to_stft(mel)
    audio = librosa.griffinlim(stft, n_iter=n_iters)
    return audio

async def generate_sung_audio_with_background(text_str, output_filename="sung_output.wav"):
    try:
        # Validar texto
        if not text_str or not text_str.strip():
            raise ValueError("El texto proporcionado está vacío.")

        # Procesar texto a IDs
        text = torch.tensor([ord(c) % VOCAB_SIZE for c in text_str]).unsqueeze(0)
        seq_len = text.size(1)
        pitches = torch.randint(60, 84, (1, seq_len))  # Pitches aleatorios

        # Modelo TTS
        model = SingingTTS()
        mel = model(text, pitches)[0]  # [seq_len, MEL_BINS]

        # Vocoder a waveform
        voice_audio = griffin_lim(mel, n_iters=10)

        # Generar música
        music_model = MusicGenerator()
        freqs = music_model(seq_len)
        background_audio = generate_waveform(freqs, len(voice_audio) * 1000 / SAMPLE_RATE)

        # Mezclar
        min_len = min(len(voice_audio), len(background_audio))
        combined = voice_audio[:min_len] * 0.7 + background_audio[:min_len] * 0.3

        # Guardar en memoria
        output_io = BytesIO()
        sf.write(output_io, combined, SAMPLE_RATE, format='wav')
        output_io.seek(0)
        return output_io.getvalue()

    except Exception as e:
        raise Exception(f"Error al generar el audio: {str(e)}")

async def main():
    # Leer archivo .txt
    try:
        with open("output.txt", "r", encoding="utf-8") as file:
            sample_text = file.read()
    except FileNotFoundError:
        sample_text = "Este es un texto de ejemplo para cantar."
        print("Archivo 'texto.txt' no encontrado, usando texto de ejemplo.")

    try:
        output_audio = await generate_sung_audio_with_background(sample_text)
        with open("sung_output.wav", "wb") as f:
            f.write(output_audio)
        print("Audio generado en sung_output.wav.")
    except Exception as e:
        print(f"Error: {str(e)}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())