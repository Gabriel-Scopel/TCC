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

# --- Etapa 1: Carregar e preparar os dados ---
with open('dadosPT.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Usa original_grade como target; se não existir, preenche com 0
df['original_grade'] = df.get('original_grade', 0)
df['original_grade'] = df['original_grade'].fillna(0)
df['target'] = df['original_grade']

# (Opcional) Balanceia a base duplicando exemplos com nota 0 e 5
df_zeros = df[df['target'] == 0]
df_fives = df[df['target'] == 5]
df = pd.concat([df, df_zeros, df_fives], ignore_index=True)

# Define as features e o target
X = df[['elmo_grade', 'use_grade', 'bert_grade']].values
Y = df['target'].values.reshape(-1, 1)

# Escalonamento das features para melhorar a convergência
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# --- Etapa 2: Construir o modelo MLP avançado ---
inputs = keras.Input(shape=(3,))
x = layers.BatchNormalization()(inputs)
x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense1')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense2')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense3')(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation='linear', name='saida')(x)
# Garante que a saída fique no intervalo [0, 5]
output_clipped = layers.Lambda(lambda t: tf.clip_by_value(t, 0.0, 5.0), name='clip')(output)

model = keras.Model(inputs, output_clipped)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])

model.summary()

# --- Etapa 3: Treinamento com callbacks avançados ---
checkpoint = keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss',
                                               save_best_only=True, verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=10, min_lr=1e-6, verbose=1)

history = model.fit(X_train, Y_train, epochs=300, batch_size=8,
                    validation_data=(X_test, Y_test),
                    callbacks=[checkpoint, reduce_lr])

# --- Etapa 4: Carregar os melhores pesos e avaliar o modelo ---
model.load_weights('best_model.keras')
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

# --- Etapa 5: Estimar a importância de cada embedding ---
# Aqui os pesos da camada 'dense1' são usados como proxy.
w_dense1 = model.get_layer('dense1').get_weights()[0]  # forma: (3, 64)
importance = np.mean(np.abs(w_dense1), axis=1)
importance_norm = importance / np.sum(importance)
print("Importância normalizada dos embeddings (elmo, use, bert):", importance_norm)

pesos_dict = {
    'elmo': float(importance_norm[0]),
    'use': float(importance_norm[1]),
    'bert': float(importance_norm[2])
}
with open('pesosPT_MLP.json', 'w', encoding='utf-8') as f:
    json.dump(pesos_dict, f, indent=4)

# --- Etapa 6: Cálculo da nota do sistema por média ponderada ---
# Em vez de usar a predição do modelo, calcula-se a média ponderada para cada questão
# usando as notas originais (não escalonadas) e os pesos determinados.
with open('pesosPT_MLP.json', 'r', encoding='utf-8') as f:
    pesos = json.load(f)

df['nota_do_sistema'] = (
    df['elmo_grade'] * pesos['elmo'] +
    df['use_grade'] * pesos['use'] +
    df['bert_grade'] * pesos['bert']
)

# Se desejar, pode-se reescalar essa média para o intervalo [0, 5] usando os limites dos targets
min_ref = df['target'].min()
max_ref = df['target'].max()
df['nota_do_sistema'] = ((df['nota_do_sistema'] - min_ref) / (max_ref - min_ref)) * 5
df['nota_do_sistema'] = df['nota_do_sistema'].clip(lower=0, upper=5)

output_records = df.to_dict(orient='records')
with open('media_PT_MLP.json', 'w', encoding='utf-8') as f:
    json.dump(output_records, f, ensure_ascii=False, indent=4)

print("Arquivo gerado: media_PT_MLP.json")
print("Pesos salvos em: pesosPT_MLP.json")

plt.figure(figsize=(8, 6))
plt.scatter(df['target'], df['nota_do_sistema'], alpha=0.6, edgecolors='k')
plt.plot([0, 5], [0, 5], 'r--', label='Ideal (y = x)')
plt.title('Comparação: Nota do Professor vs Média Ponderada Calculada')
plt.xlabel('Nota do Professor (target)')
plt.ylabel('Média Ponderada (nota_do_sistema)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
