import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Carregar o modelo e o tokenizador do BERT pré-treinado
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# Função para processar texto com BERT e obter embeddings
def obter_embedding(texto):
    """Gera embeddings para o texto usando BERT."""
    inputs = tokenizer(texto, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()  # Média dos embeddings das palavras

# Função para calcular a similaridade de cosseno
def calcular_similaridade(resposta_aluno, respostas_referencia):
    """Calcula a similaridade de cosseno entre a resposta do aluno e as respostas de referência."""
    aluno_vec = obter_embedding(resposta_aluno)
    referencia_vecs = np.array([obter_embedding(resp) for resp in respostas_referencia])
    
    similaridades = cosine_similarity(aluno_vec, referencia_vecs.reshape(len(respostas_referencia), -1))[0]
    return max(similaridades)  # Retorna a maior similaridade encontrada

# Carregar os dados
input_path = "D:/Downloads/ProjTCCMurilo/dataset/BasesTratadas/ptbrData_0a5.json"
output_path = "D:/Downloads/ProjTCCMurilo/correcao_bert.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

output_data = []

# Processar cada questão e resposta
for item in data:
    numero_pergunta = item["number_question"]
    respostas_referencia = [resp["reference_response"] for resp in item["reference_responses"]]
    
    for resposta_aluno in item["responses_students"]:
        resposta_texto = resposta_aluno["answer_question"]
        nota_original = resposta_aluno["grade"]
        
        # Calcular similaridade
        similaridade = calcular_similaridade(resposta_texto, respostas_referencia)
        
        output_data.append({
            "number_question": numero_pergunta,
            "answer_question": resposta_texto,
            "original_grade": nota_original,
            "bert_similarity": similaridade
        })

# Salvar o resultado em um arquivo JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Correção concluída. Resultados salvos em '{output_path}'")
