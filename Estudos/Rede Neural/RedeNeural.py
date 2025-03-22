import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Função para carregar JSON
def carregar_json(caminho):
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

# Caminhos dos arquivos JSON
caminho_bert = "bert_graded_responses.json"
caminho_elmo = "correcao_elmo_EN.json"
caminho_use = "correcao_use_EN.json"

bert_data = carregar_json(caminho_bert)
elmo_data = carregar_json(caminho_elmo)
use_data = carregar_json(caminho_use)

# Indexar respostas para cada modelo
def indexar_respostas_bert(dados):
    index = {}
    for questao in dados:
        for resposta in questao["responses_students"]:
            key = (questao["number_question"], resposta["answer_question"])
            index[key] = (resposta["bert_grade"], resposta["grade"])  # Nota do corretor também
    return index

def indexar_respostas_generico(dados, modelo):
    index = {}
    for resposta in dados:
        key = (resposta["number_question"], resposta["answer_question"])
        index[key] = resposta[f"{modelo}_grade"]
    return index

bert_index = indexar_respostas_bert(bert_data)
elmo_index = indexar_respostas_generico(elmo_data, "elmo")
use_index = indexar_respostas_generico(use_data, "use")

# Criar dataset combinando as notas
X = []
y = []

for key, (bert_grade, human_grade) in bert_index.items():
    if key in elmo_index and key in use_index:
        elmo_grade = elmo_index[key]
        use_grade = use_index[key]
        
        # Criar features extras
        mean_grade = np.mean([bert_grade, elmo_grade, use_grade])
        variance_grade = np.var([bert_grade, elmo_grade, use_grade])  # Nova feature

        X.append([bert_grade, elmo_grade, use_grade, mean_grade, variance_grade])
        y.append(human_grade)

# Converter para arrays numpy
X = np.array(X)
y = np.array(y).reshape(-1, 1)  # Garantir que y seja uma matriz coluna

# Normalizar os dados
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar modelo de rede neural com ajustes
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(5,)),  # Aumento de neurônios e uso de ReLU
    keras.layers.Dropout(0.3),  # Aumento do dropout
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="linear")  # Saída contínua
])

# Compilar o modelo com um otimizador diferente e taxa de aprendizado ajustada
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae", metrics=["mae"])

# Treinar o modelo
model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), batch_size=16, verbose=1)

# Avaliação
loss, mae = model.evaluate(X_test, y_test)
mae_original = scaler_y.inverse_transform([[mae]])[0][0]  # Convertendo para escala original
print(f"Erro Médio Absoluto (MAE): {mae_original:.2f}")

# Cálculo do RMSE
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_original = scaler_y.inverse_transform([[rmse]])[0][0]  # Convertendo para escala original
print(f"Erro Quadrático Médio (RMSE): {rmse_original:.2f}")
