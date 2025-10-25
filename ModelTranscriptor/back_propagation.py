import numpy as np
import pandas as pd
import re

# ===== 1. Tokenización básica =====
def simple_tokenizer(text):
    return re.findall(r"[a-záéíóúñü]+", text.lower())

# Cargar dataset
df = pd.read_csv("lyrics_dataset.csv")
df["english_tokens"] = df["english"].apply(simple_tokenizer)
df["spanish_tokens"] = df["spanish"].apply(simple_tokenizer)

# ===== 2. Construir pares palabra a palabra =====
pairs = []
for eng, spa in zip(df["english_tokens"], df["spanish_tokens"]):
    for e, s in zip(eng, spa):  # ⚠️ toma solo hasta el mínimo largo
        pairs.append((e, s))

X_words = [p[0] for p in pairs]
y_words = [p[1] for p in pairs]

# ===== 3. Vocabularios =====
vocab_in = sorted(list(set(X_words)))
vocab_out = sorted(list(set(y_words)))

def one_hot_encode(vocab, word):
    vector = np.zeros(len(vocab))
    vector[vocab.index(word)] = 1
    return vector

def one_hot_decode(vocab, vector):
    return vocab[np.argmax(vector)]

# Conversión a One-Hot
X = np.array([one_hot_encode(vocab_in, w) for w in X_words])
y = np.array([one_hot_encode(vocab_out, w) for w in y_words])

# ===== 4. Inicializar Red Neuronal =====
np.random.seed(42)
input_neurons = len(vocab_in)
hidden_neurons = 16   # puedes ajustar
output_neurons = len(vocab_out)

# W1 = np.random.uniform(-0.1, 1, (input_neurons, hidden_neurons))
b1 = np.zeros((1, hidden_neurons))
# W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
b2 = np.zeros((1, output_neurons))

limit1 = np.sqrt(6 / (input_neurons + hidden_neurons))
W1 = np.random.uniform(-limit1, limit1, (input_neurons, hidden_neurons))

limit2 = np.sqrt(6 / (hidden_neurons + output_neurons))
W2 = np.random.uniform(-limit2, limit2, (hidden_neurons, output_neurons))
 
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - x ** 2

# ===== 5. Entrenamiento =====
learning_rate = 0.5
min_error = 1e-4
max_epochs = 20000

for epoch in range(max_epochs):
    # Forward
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = tanh(z2)

    # Error
    error = y - a2
    error_value = np.mean(np.square(error))

    # Backpropagation
    d_a2 = error * tanh_derivative(a2)
    d_a1 = d_a2.dot(W2.T) * tanh_derivative(a1)

    W2 += a1.T.dot(d_a2) * learning_rate
    b2 += np.sum(d_a2, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_a1) * learning_rate
    b1 += np.sum(d_a1, axis=0, keepdims=True) * learning_rate

    if epoch % 2000 == 0:
        print(f"Época {epoch}, Error: {error_value:.6f}")

# ===== 6. Resultados de traducción =====
print("\nResultados de traducción (demo):")
for i, word in enumerate(X_words[:20]):  # muestra primeras 20
    pred_vector = a2[i]
    pred_word = one_hot_decode(vocab_out, pred_vector)
    print(f"{word} -> {pred_word}")
