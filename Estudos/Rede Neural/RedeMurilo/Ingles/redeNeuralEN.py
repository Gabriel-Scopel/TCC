import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# --- Função de conversão para dpseek_grade e dpseek_similarity ---
def convert_to_float(x):
    if isinstance(x, dict):
        try:
            return float(x.get('score', 0))
        except Exception:
            return np.nan
    else:
        try:
            return float(x)
        except Exception:
            return np.nan

# --- Etapa 1: Carregar e preparar os dados ---
with open('dadosEN.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Converter os campos numéricos para float (notas e similaridades)
cols_num = ['elmo_grade', 'use_grade', 'bert_grade', 'dpseek_grade',
            'elmo_similarity', 'use_similarity', 'bert_similarity', 'dpseek_similarity', 'original_grade']
for col in cols_num:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Preenche NaN com 0 (você pode ajustar esse valor se necessário)
df = df.fillna(0)

# Usa original_grade como target
df['target'] = df['original_grade']

# Balanceia a base duplicando exemplos com nota 0 e 5 (opcional)
df_zeros = df[df['target'] == 0]
df_fives = df[df['target'] == 5]
df = pd.concat([df, df_zeros, df_fives], ignore_index=True)

# Define as features: agora com 4 inputs (notas dos 4 embeddings)
X = df[['elmo_similarity', 'use_similarity', 'bert_similarity', 'dpseek_similarity']].values
Y = df['target'].values.reshape(-1, 1)

# Escalonamento das features (para treinamento)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão em treino (70%) e teste (30%)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# --- Etapa 2: Construir o modelo MLP avançado (4 inputs) ---
inputs = keras.Input(shape=(4,))
x = layers.BatchNormalization()(inputs)
x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense1')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense2')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense3')(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation='linear', name='saida')(x)
output_clipped = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 5.0), name='clip')(output)

# Utilize o clipnorm para evitar gradientes explosivos
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

model = keras.Model(inputs, output_clipped)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])

model.summary()

# --- Etapa 3: Treinamento com callbacks avançados ---
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model_new.weights.h5', monitor='val_loss',
    save_best_only=True, save_weights_only=True, verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=10, min_lr=1e-6, verbose=1)

history = model.fit(X_train, Y_train, epochs=300, batch_size=8,
                    validation_data=(X_test, Y_test),
                    callbacks=[checkpoint, reduce_lr])

# --- Etapa 4: Carregar os melhores pesos e avaliar o modelo ---
if os.path.exists('best_model_new.weights.h5'):
    model.load_weights('best_model_new.weights.h5')
else:
    print("Arquivo de pesos não encontrado. Utilizando os pesos atuais do modelo.")

loss, rmse, mae = model.evaluate(X_test, Y_test)
print('Test Loss (MSE):', loss)
print("RMSE:", rmse)
print("MAE:", mae)
y_pred = model.predict(X_test)
print("R²:", r2_score(Y_test, y_pred))

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Loss Treino')
plt.plot(history.history['val_loss'], label='Loss Validação')
plt.title('Evolução da Loss')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid()
plt.show()

# --- Etapa 5: Estimar a importância dos embeddings (proxy) ---
# Obter os pesos da camada 'dense1' (4 entradas)
w_dense1 = model.get_layer('dense1').get_weights()[0]  # shape: (4, 64)
importance = np.mean(np.abs(w_dense1), axis=1)
sum_importance = np.sum(importance)
if sum_importance == 0:
    importance_norm = np.full_like(importance, 1.0 / importance.shape[0])
else:
    importance_norm = importance / sum_importance

print("Importância normalizada (proxy) dos embeddings (elmo, use, bert, dpseek):", importance_norm)
pesos_dict = {
    'elmo': float(importance_norm[0]),
    'use': float(importance_norm[1]),
    'bert': float(importance_norm[2]),
    'dpseek': float(importance_norm[3])
}
with open('pesosEN_MLP.json', 'w', encoding='utf-8') as f:
    json.dump(pesos_dict, f, indent=4)

# --- Etapa 6: Cálculo da nota do sistema usando similaridade de cosseno (4 pesos) ---
def calcular_media_ponderada(row):
    sim_dpseek = convert_to_float(row['dpseek_similarity'])
    sim_total = (row['elmo_similarity'] + row['use_similarity'] +
                 row['bert_similarity'] + sim_dpseek)
    if sim_total > 0:
        peso_elmo = row['elmo_similarity'] / sim_total
        peso_use = row['use_similarity'] / sim_total
        peso_bert = row['bert_similarity'] / sim_total
        peso_dpseek = sim_dpseek / sim_total
    else:
        peso_elmo = peso_use = peso_bert = peso_dpseek = 0.25
    return (row['elmo_grade'] * peso_elmo +
            row['use_grade'] * peso_use +
            row['bert_grade'] * peso_bert +
            row['dpseek_grade'] * peso_dpseek)

df['nota_do_sistema'] = df.apply(calcular_media_ponderada, axis=1)

# (Opcional) Reescalonar a nota para o intervalo [0, 5] com base nos limites do target
min_ref = df['target'].min()
max_ref = df['target'].max()
df['nota_do_sistema'] = ((df['nota_do_sistema'] - min_ref) / (max_ref - min_ref)) * 5
df['nota_do_sistema'] = df['nota_do_sistema'].clip(lower=0, upper=5)

output_records = df.to_dict(orient='records')
with open('media_EN_MLP.json', 'w', encoding='utf-8') as f:
    json.dump(output_records, f, ensure_ascii=False, indent=4)

print("Arquivo gerado: media_EN_MLP.json")
print("Pesos (do proxy da rede) salvos em: pesosEN_MLP.json")

plt.figure(figsize=(8, 6))
plt.scatter(df['target'], df['nota_do_sistema'], alpha=0.6, edgecolors='k')
plt.plot([0, 5], [0, 5], 'r--', label='Ideal (y = x)')
plt.title('Comparação: Nota do Professor vs Nota do Sistema (4 Embeddings)')
plt.xlabel('Nota do Professor (target)')
plt.ylabel('Nota do Sistema (nota_do_sistema)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
