# predict.py
import re
import torch
import torch.nn as nn
import argparse
from pathlib import Path

# Clases del modelo (las mismas que en tu código original)
class Vocab:
    def __init__(self, tokens_list=None, min_freq=1, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        if tokens_list is not None:
            self.freq = {}
            for tokens in tokens_list:
                for t in tokens:
                    self.freq[t] = self.freq.get(t, 0) + 1
            toks = [t for t, f in self.freq.items() if f >= min_freq]
            toks = sorted(toks)
            self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"] + reserved_tokens + toks
        else:
            self.itos = []
        self.stoi = {t: i for i, t in enumerate(self.itos)} if self.itos else {}
    
    def __len__(self):
        return len(self.itos)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.W1 = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        self.W2 = nn.Linear(dec_hid_dim, dec_hid_dim, bias=False)
        self.V = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden, mask=None):
        dec_hidden = hidden[-1].unsqueeze(1)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(dec_hidden)))
        score = score.squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.attention = attention
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers=n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(input_step).unsqueeze(1))
        context, attn_weights = self.attention(encoder_outputs, hidden, mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        pred_input = torch.cat([output, context, embedded.squeeze(1)], dim=1)
        pred = self.fc_out(pred_input)
        return pred, hidden, attn_weights

# Funciones de utilidad
def simple_tokenizer(text: str):
    return re.findall(r"[a-záéíóúñü]+", str(text).lower())

def create_mask(src_pad):
    return (src_pad != 0).to(src_pad.device)

def translate_sentence(sentence, src_vocab, trg_vocab, encoder, decoder, device, max_len=40):
    encoder.eval()
    decoder.eval()
    
    # Tokenizar la oración
    tokens = simple_tokenizer(sentence)
    src_idx = [src_vocab.stoi.get(t, src_vocab.stoi["<unk>"]) for t in tokens]
    
    src_tensor = torch.tensor(src_idx, dtype=torch.long).unsqueeze(0).to(device)
    src_len = torch.tensor([len(src_idx)]).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = encoder(src_tensor, src_len)
        mask = create_mask(src_tensor)
        input_tok = torch.tensor([trg_vocab.stoi["<sos>"]], dtype=torch.long).to(device)
        
        result_tokens = []
        for _ in range(max_len):
            output, hidden, attn = decoder(input_tok, hidden, encoder_outputs, mask)
            top1 = output.argmax(1).item()
            if top1 == trg_vocab.stoi["<eos>"]:
                break
            result_tokens.append(trg_vocab.itos[top1])
            input_tok = torch.tensor([top1], dtype=torch.long).to(device)
    
    return " ".join(result_tokens)

def load_model(model_path, device):
    """Carga el modelo entrenado"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Crear objetos Vocab a partir del estado guardado
    src_vocab = Vocab()
    src_vocab.itos = checkpoint['src_vocab']['itos']
    src_vocab.stoi = checkpoint['src_vocab']['stoi']
    
    trg_vocab = Vocab()
    trg_vocab.itos = checkpoint['trg_vocab']['itos']
    trg_vocab.stoi = checkpoint['trg_vocab']['stoi']
    
    # Obtener configuración del modelo
    config = checkpoint.get('model_config', {
        'emb_dim': 128,
        'hid_dim': 256,
        'n_layers': 1,
        'dropout': 0.2
    })
    
    # Crear modelos
    attention = BahdanauAttention(
        enc_hid_dim=config['hid_dim'],
        dec_hid_dim=config['hid_dim']
    )
    
    encoder = Encoder(
        input_dim=len(src_vocab),
        emb_dim=config['emb_dim'],
        hid_dim=config['hid_dim'],
        n_layers=config['n_layers'],
        dropout=config['dropout']
    )
    
    decoder = Decoder(
        output_dim=len(trg_vocab),
        emb_dim=config['emb_dim'],
        hid_dim=config['hid_dim'],
        attention=attention,
        n_layers=config['n_layers'],
        dropout=config['dropout']
    )
    
    # Cargar pesos
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    
    # Mover a dispositivo
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    attention = attention.to(device)
    
    return encoder, decoder, src_vocab, trg_vocab

def main():
    parser = argparse.ArgumentParser(description="Traducir texto usando modelo seq2seq entrenado")
    parser.add_argument("--model_path", type=str, required=True, help="best_seq2seq.pt")
    parser.add_argument("--text", type=str, help="Texto a traducir (opcional)")
    parser.add_argument("--input_file", type=str, help="transcripcion_20250823_111904.txt")
    parser.add_argument("--output_file", type=str, help="result.txt")
    parser.add_argument("--use_gpu", action="store_true", help="Usar GPU si está disponible")
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelo
    print(f"Cargando modelo desde: {args.model_path}")
    try:
        encoder, decoder, src_vocab, trg_vocab = load_model(args.model_path, device)
        print("Modelo cargado exitosamente")
        print(f"Tamaño vocabulario fuente: {len(src_vocab)}")
        print(f"Tamaño vocabulario objetivo: {len(trg_vocab)}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de que el modelo fue guardado con la versión corregida del código de entrenamiento")
        return
    
    # Obtener texto para traducir
    texts_to_translate = []
    
    if args.text:
        texts_to_translate.append(args.text)
    
    if args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts_to_translate.append(line)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {args.input_file}")
            return
    
    # Si no se proporcionó texto, modo interactivo
    if not texts_to_translate:
        print("Modo interactivo. Escribe 'quit' para salir.")
        while True:
            text = input("\nTexto en inglés: ").strip()
            if text.lower() in ['quit', 'exit', 'salir']:
                break
            if text:
                texts_to_translate.append(text)
    
    # Realizar traducciones
    results = []
    for text in texts_to_translate:
        try:
            translation = translate_sentence(text, src_vocab, trg_vocab, encoder, decoder, device)
            results.append(f"Original: {text}")
            results.append(f"Traducción: {translation}")
            results.append("---")
            
            print(f"\nOriginal: {text}")
            print(f"Traducción: {translation}")
            print("---")
        except Exception as e:
            print(f"Error traduciendo: {text}")
            print(f"Error: {e}")
    
    # Guardar resultados si se especificó archivo de salida
    if args.output_file and results:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(results))
            print(f"\nResultados guardados en: {args.output_file}")
        except Exception as e:
            print(f"Error guardando resultados: {e}")

if __name__ == "__main__":
    main()