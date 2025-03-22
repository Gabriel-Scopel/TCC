import json
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Carregar o modelo Universal Sentence Encoder (USE)
print("Carregando modelo USE...")
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("Modelo USE carregado com sucesso!")

# Caminhos dos arquivos de entrada e saída
input_path = "D:\Downloads\ProjTCCMurilo\dataset\BasesTratadas\esData.json"
output_path = "D:\Downloads\ProjTCCMurilo\correcao_use_ES.json"

# Carregar os dados da base tratada
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def calcular_similaridade(resposta_aluno, respostas_referencia):
    """Calcula a similaridade de cosseno entre a resposta do aluno e as respostas de referência."""
    embeddings = model([resposta_aluno] + respostas_referencia)
    aluno_vec = embeddings[0].numpy().reshape(1, -1)
    referencias_vec = embeddings[1:].numpy()
    
    # Calcular a similaridade com cada resposta de referência
    similaridades = cosine_similarity(aluno_vec, referencias_vec)[0]
    return max(similaridades)  # Retorna a maior similaridade encontrada

def mapear_nota(similarity, min_sim=0.0, max_sim=1.0, min_score=0.0, max_score=5.0):
    """Mapeia a similaridade de cosseno para um range de notas (0 a 5)."""
    return min_score + (max_score - min_score) * ((similarity - min_sim) / (max_sim - min_sim))

output_data = []

total_questoes = len(data)
print(f"Processando {total_questoes} questões...")

for item in data:
    numero_pergunta = item["number_question"]
    respostas_referencia = [resp["reference_response"] for resp in item["reference_responses"]]
    
    for resposta_aluno in item["responses_students"]:
        resposta_texto = resposta_aluno["answer_question"]
        nota_original = resposta_aluno["grade"]
        
        # Calcular similaridade e converter para nota
        similaridade = calcular_similaridade(resposta_texto, respostas_referencia)
        nota_corrigida = mapear_nota(similaridade)
        
        output_data.append({
            "number_question": int(numero_pergunta),
            "answer_question": resposta_texto,
            "original_grade": float(nota_original),
            "use_similarity": float(similaridade),
            "use_grade": round(float(nota_corrigida), 2)
        })

# Salvar o resultado em um arquivo JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Correção concluída. Resultados salvos em '{output_path}'")