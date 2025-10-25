import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar el dataset
df = pd.read_csv('data2.csv')

# Limpiar y preprocesar el texto
def preprocess_text(text):
    text = text.lower()
    text = text.replace('"', '')
    return text.strip()

df['English'] = df['English'].apply(preprocess_text)
df['Spanish'] = df['Spanish'].apply(preprocess_text)

# Tokenización para inglés
eng_tokenizer = Tokenizer(filters='', oov_token='<OOV>')
eng_tokenizer.fit_on_texts(df['English'])
eng_sequences = eng_tokenizer.texts_to_sequences(df['English'])

# Tokenización para español - AÑADIR TOKENS ESPECIALES
spa_tokenizer = Tokenizer(filters='', oov_token='<OOV>')
spa_tokenizer.fit_on_texts(df['Spanish'])

# Añadir tokens especiales al vocabulario español manualmente
spa_tokenizer.word_index['<start>'] = len(spa_tokenizer.word_index) + 1
spa_tokenizer.word_index['<end>'] = len(spa_tokenizer.word_index) + 1

spa_sequences = spa_tokenizer.texts_to_sequences(df['Spanish'])

# Añadir tokens de inicio y fin a las secuencias españolas
spa_sequences_with_tokens = []
for seq in spa_sequences:
    spa_sequences_with_tokens.append([spa_tokenizer.word_index['<start>']] + seq + [spa_tokenizer.word_index['<end>']])

# Encontrar la longitud máxima
max_length_eng = max(len(seq) for seq in eng_sequences)
max_length_spa = max(len(seq) for seq in spa_sequences_with_tokens)
max_length = max(max_length_eng, max_length_spa)

print(f"Longitud máxima necesaria: {max_length}")

# Padding para igualar longitudes
eng_padded = pad_sequences(eng_sequences, maxlen=max_length, padding='post')
spa_padded = pad_sequences(spa_sequences_with_tokens, maxlen=max_length, padding='post')

# Vocabularios
eng_vocab_size = len(eng_tokenizer.word_index) + 1
spa_vocab_size = len(spa_tokenizer.word_index) + 1

print(f"Tamaño vocabulario inglés: {eng_vocab_size}")
print(f"Tamaño vocabulario español: {spa_vocab_size}")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input

# Hiperparámetros
embedding_dim = 128  # Reducido para dataset pequeño
latent_dim = 256     # Reducido para dataset pequeño

# Encoder
encoder_inputs = Input(shape=(max_length,))
enc_emb = Embedding(eng_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder - ¡CORREGIDO! Ahora usa max_length-1
decoder_inputs = Input(shape=(max_length-1,))  # Cambiado a max_length-1
dec_emb = Embedding(spa_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(spa_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Modelo
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Preparar datos para entrenamiento - ¡CORREGIDO!
decoder_input_data = spa_padded[:, :-1]  # Todos excepto el último token
decoder_target_data = spa_padded[:, 1:]   # Todos excepto el primer token

# Verificar dimensiones
print(f"Forma de eng_padded: {eng_padded.shape}")
print(f"Forma de decoder_input_data: {decoder_input_data.shape}")
print(f"Forma de decoder_target_data: {decoder_target_data.shape}")

# Asegurar que las dimensiones coincidan
assert eng_padded.shape[0] == decoder_input_data.shape[0] == decoder_target_data.shape[0]
assert decoder_input_data.shape[1] == max_length - 1
assert decoder_target_data.shape[1] == max_length - 1

# Entrenamiento con menos épocas y callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = model.fit(
    [eng_padded, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=16,  # Batch más pequeño
    epochs=250,      # Menos épocas
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stopping]
)

# Modelo de encoder para inferencia
encoder_model = Model(encoder_inputs, encoder_states)

# Modelo de decoder para inferencia - ¡CORREGIDO!
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Usar una nueva entrada para el decoder en inferencia
decoder_inputs_inference = Input(shape=(1,))  # Solo un paso a la vez
dec_emb_inference = Embedding(spa_vocab_size, embedding_dim)(decoder_inputs_inference)
decoder_outputs_inference, state_h_inference, state_c_inference = decoder_lstm(
    dec_emb_inference, initial_state=decoder_states_inputs
)
decoder_outputs_inference = decoder_dense(decoder_outputs_inference)

decoder_model = Model(
    [decoder_inputs_inference] + decoder_states_inputs,
    [decoder_outputs_inference] + [state_h_inference, state_c_inference]
)

# Función para predecir - CORREGIDA
def predict_spanish(english_text):
    # Preprocesar
    processed_text = preprocess_text(english_text)
    
    # Tokenizar y padding
    seq = eng_tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    
    # Estados del encoder
    states_value = encoder_model.predict(padded, verbose=0)
    
    # Generar secuencia objetivo
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = spa_tokenizer.word_index['<start>']
    
    stop_condition = False
    decoded_sentence = []
    max_decoder_steps = 20  # Límite para evitar loops infinitos
    
    for t in range(max_decoder_steps):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )
        
        # Obtener el token predicho
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = spa_tokenizer.index_word.get(sampled_token_index, '<OOV>')
        
        if sampled_word == '<end>' or sampled_token_index == 0:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)
        
        # Actualizar el target sequence y estados
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
        
        if stop_condition:
            break
    
    return ' '.join(decoded_sentence)

# Probar el modelo
english_lyric = "When marimba rhythms start to play"
spanish_translation = predict_spanish(english_lyric)
print(f"English: {english_lyric}")
print(f"Spanish: {spanish_translation}")

# Probar con más ejemplos
test_phrases = [
    "Dance with me, make me sway",
    "Like a lazy ocean hugs the shore", 
    "Dancin' is what clears my soul",
    "Dancin' is what to do"
]

for phrase in test_phrases:
    translation = predict_spanish(phrase)
    print(f"'{phrase}' -> '{translation}'")
    
model.save('modelo/modelo_traduccion.h5')

# Guardar los tokenizers
import pickle

with open('modelo/eng_tokenizer.pkl', 'wb') as f:
    pickle.dump(eng_tokenizer, f)
    
with open('modelo/spa_tokenizer.pkl', 'wb') as f:
    pickle.dump(spa_tokenizer, f)

# Guardar los parámetros importantes
model_info = {
    'max_length': max_length,
    'embedding_dim': embedding_dim,
    'latent_dim': latent_dim,
    'eng_vocab_size': eng_vocab_size,
    'spa_vocab_size': spa_vocab_size
}

with open('modelo/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("Modelo y tokenizers guardados exitosamente")

def guardar_modelo_completo(model, eng_tokenizer, spa_tokenizer, max_length, 
                           embedding_dim, latent_dim, eng_vocab_size, spa_vocab_size):
    """Guardar todo lo necesario para inferencia"""
    
    # 1. Guardar modelo principal
    model.save('modelo/modelo_traduccion.h5')
    
    # 2. Guardar tokenizers
    import pickle
    with open('modelo/eng_tokenizer.pkl', 'wb') as f:
        pickle.dump(eng_tokenizer, f)
    with open('modelo/spa_tokenizer.pkl', 'wb') as f:
        pickle.dump(spa_tokenizer, f)
    
    # 3. Guardar parámetros
    model_info = {
        'max_length': max_length,
        'embedding_dim': embedding_dim,
        'latent_dim': latent_dim,
        'eng_vocab_size': eng_vocab_size,
        'spa_vocab_size': spa_vocab_size
    }
    with open('modelo/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    # 4. Guardar modelos de inferencia
    encoder_model.save('modelo/encoder_model.h5')
    decoder_model.save('modelo/decoder_model.h5')
    
    print("✅ Todos los modelos guardados exitosamente")

# Llama esta función después de entrenar
guardar_modelo_completo(model, eng_tokenizer, spa_tokenizer, max_length,
                       embedding_dim, latent_dim, eng_vocab_size, spa_vocab_size)

import matplotlib.pyplot as plt
import numpy as np

# Crear la gráfica de entrenamiento
plt.figure(figsize=(12, 5))

# Gráfica de pérdida (loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento', linewidth=2)
plt.plot(history.history['val_loss'], label='Pérdida de validación', linewidth=2)
plt.title('Evolución de la Pérdida durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica de precisión (accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Precisión de validación', linewidth=2)
plt.title('Evolución de la Precisión durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True, alpha=0.3)

# Ajustar layout y guardar
plt.tight_layout()
plt.savefig('entrenamiento_modelo.png', dpi=300, bbox_inches='tight')
plt.show()

# Mostrar métricas finales
print(f"\nMétricas finales del entrenamiento:")
print(f"Pérdida final de entrenamiento: {history.history['loss'][-1]:.4f}")
print(f"Pérdida final de validación: {history.history['val_loss'][-1]:.4f}")
print(f"Precisión final de entrenamiento: {history.history['accuracy'][-1]:.4f}")
print(f"Precisión final de validación: {history.history['val_accuracy'][-1]:.4f}")

# Calcular épocas de entrenamiento
epochs_trained = len(history.history['loss'])
print(f"\nÉpocas entrenadas: {epochs_trained}")