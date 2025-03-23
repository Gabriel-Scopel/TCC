import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# --- Etapa 1: Carregar e preparar os dados ---

# Carrega o JSON unificado
with open('dados_unificados.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Converte para DataFrame
df = pd.DataFrame(data)

# Usa original_grade como target, se existir; senão, 0
df['original_grade'] = df.get('original_grade', 0)
df['original_grade'] = df['original_grade'].fillna(0)
df['target'] = df['original_grade']

# Balanceia a base duplicando exemplos com nota 0 e 5
df_zeros = df[df['target'] == 0]
df_cincos = df[df['target'] == 5]
df = pd.concat([df, df_zeros, df_cincos], ignore_index=True)

# Define entradas e saídas
X = df[['elmo_grade', 'use_grade', 'bert_grade']].values
Y = df['target'].values.reshape(-1, 1)

# Divide em treino/teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# --- Etapa 2: Camada customizada ---

class WeightedAverageLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedAverageLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.p = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
            name='pesos',
            regularizer=regularizers.l2(0.01)
        )
        super(WeightedAverageLayer, self).build(input_shape)

    def call(self, inputs):
        numerator = tf.reduce_sum(inputs * self.p, axis=-1, keepdims=True)
        denominator = tf.reduce_sum(self.p)
        return numerator / denominator

# --- Etapa 3: Modelo ---

inputs = keras.Input(shape=(3,))
outputs = WeightedAverageLayer()(inputs)
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_absolute_error',
              metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()])
model.summary()

# --- Etapa 4: Treinamento ---
history = model.fit(X_train, Y_train, epochs=200, batch_size=8, validation_data=(X_test, Y_test))

# --- Etapa 5: Avaliação ---
loss, rmse, mae = model.evaluate(X_test, Y_test)
print('Test Loss (MAE):', loss)
print("RMSE:", rmse)
print("MAE:", mae)

# Predição
y_pred = model.predict(X_test)
print("R²:", r2_score(Y_test, y_pred))

# Gráfico da loss
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Evolução da Loss')
plt.xlabel('Epochs')
plt.ylabel('Erro')
plt.legend()
plt.grid()
plt.show()

# --- Etapa 6: Pesos aprendidos e nota final ---
pesos = model.layers[-1].get_weights()[0]
soma_pesos = np.sum(pesos)

# Aplica média ponderada
df['nota_do_sistema'] = (
    df['elmo_grade'] * pesos[0] +
    df['use_grade'] * pesos[1] +
    df['bert_grade'] * pesos[2]
) / soma_pesos

# Escala para 0-5 usando target como referência
min_ref = df['target'].min()
max_ref = df['target'].max()
df['nota_do_sistema'] = ((df['nota_do_sistema'] - min_ref) / (max_ref - min_ref)) * 5

# --- Etapa 7: Exportação ---
output = df.to_dict(orient='records')
with open('dados_com_media_ponderada.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

# Exporta os pesos
pesos_dict = {
    'elmo': float(pesos[0]),
    'use': float(pesos[1]),
    'bert': float(pesos[2])
}
with open('pesos_otimizados.json', 'w') as f:
    json.dump(pesos_dict, f, indent=4)

print("Arquivo gerado: dados_com_media_ponderada.json")
print("Pesos salvos em: pesos_otimizados.json")

# --- Etapa 8: Gráfico de Dispersão entre notas ---

plt.figure(figsize=(8, 6))
plt.scatter(df['original_grade'], df['nota_do_sistema'], alpha=0.6, edgecolors='k')
plt.plot([0, 5], [0, 5], 'r--', label='Ideal (y = x)')  # linha pontilhada vermelha
plt.title('Comparação: Nota do Professor vs Nota do Sistema')
plt.xlabel('Nota do Professor (original_grade)')
plt.ylabel('Nota do Sistema (nota_do_sistema)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
