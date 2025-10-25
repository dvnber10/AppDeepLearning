import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

class Traductor:
    def __init__(self):
        self.eng_tokenizer = None
        self.spa_tokenizer = None
        self.max_length = None
        self.encoder_model = None
        self.decoder_model = None
        
    def cargar_modelo(self):
        """Cargar todos los componentes del modelo"""
        try:
            # Cargar tokenizers
            with open('modelo/eng_tokenizer.pkl', 'rb') as f:
                self.eng_tokenizer = pickle.load(f)
            with open('modelo/spa_tokenizer.pkl', 'rb') as f:
                self.spa_tokenizer = pickle.load(f)
            
            # Cargar información del modelo
            with open('modelo/model_info.pkl', 'rb') as f:
                model_info = pickle.load(f)
                self.max_length = model_info['max_length']
            
            # Cargar modelos de inferencia DIRECTAMENTE
            self.encoder_model = load_model('modelo/encoder_model.h5', compile=False)
            self.decoder_model = load_model('modelo/decoder_model.h5', compile=False)

            print("✅ Modelo cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def preprocess_text(self, text):
        """Preprocesar texto"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def predecir_espanol(self, english_text):
        """Predecir traducción al español"""
        if not self.encoder_model:
            raise Exception("Modelo no cargado. Ejecuta cargar_modelo() primero.")
        
        # Preprocesar
        processed_text = self.preprocess_text(english_text)
        
        # Tokenizar y padding
        seq = self.eng_tokenizer.texts_to_sequences([processed_text])
        if not seq or not seq[0]:
            return "[Texto no reconocido]"
            
        padded = pad_sequences(seq, maxlen=self.max_length, padding='post')
        
        # Estados del encoder
        states_value = self.encoder_model.predict(padded, verbose=0)
        
        # Generar secuencia objetivo
        target_seq = np.zeros((1, 1))
        if '<start>' in self.spa_tokenizer.word_index:
            target_seq[0, 0] = self.spa_tokenizer.word_index['<start>']
        else:
            # Si no hay token <start>, usar el primer token
            target_seq[0, 0] = 1
        
        decoded_sentence = []
        max_decoder_steps = min(self.max_length, 30)  # Máximo de pasos
        
        for _ in range(max_decoder_steps):
            # Predecir con el decoder
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + list(states_value), verbose=0
            )
            
            # Obtener el token predicho
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.spa_tokenizer.index_word.get(sampled_token_index, '')
            
            # Condiciones de parada
            if (sampled_word == '<end>' or 
                sampled_token_index == 0 or 
                (self.spa_tokenizer.index_word.get(sampled_token_index) is None)):
                break
                
            if sampled_word and sampled_word not in ['<start>', '<end>', '<OOV>']:
                decoded_sentence.append(sampled_word)
                
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
        
        return ' '.join(decoded_sentence) if decoded_sentence else "[Sin traducción]"

# Función para traducir archivos
def traducir_archivo(input_file, output_file):
    """Traducir un archivo completo"""
    traductor = Traductor()
    
    if not traductor.cargar_modelo():
        print("❌ No se pudo cargar el modelo")
        return
    
    # Leer archivo
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lineas = f.read().split('\n')
    except:
        with open(input_file, 'r', encoding='latin-1') as f:
            lineas = f.read().split('\n')
    
    traducciones = []
    print(f"📝 Traduciendo {len(lineas)} líneas...")
    
    for i, linea in enumerate(lineas, 1):
        linea = linea.strip()
        if linea:
            try:
                traduccion = traductor.predecir_espanol(linea)
                resultado = f"{linea} -> {traduccion}"
                traducciones.append(traduccion)
                print(f"✅ Línea {i}: {resultado}")
            except Exception as e:
                error_msg = f"{linea} -> [Error: {str(e)}]"
                traducciones.append(error_msg)
                print(f"❌ Línea {i}: {error_msg}")
        else:
            traducciones.append("")
    
    # Guardar resultados
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(traducciones))
    
    print(f"🎉 Traducción guardada en {output_file}")

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        if not os.path.exists(input_file):
            print(f"❌ Archivo {input_file} no encontrado")
        else:
            traducir_archivo(input_file, output_file)
    else:
        print("Uso: python traductor.py archivo_entrada.txt archivo_salida.txt")
        print("Ejemplo: python traductor.py mi_cancion.txt traduccion.txt")