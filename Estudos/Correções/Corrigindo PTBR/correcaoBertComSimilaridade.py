import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Carregar modelo BERT pré-treinado para embeddings de sentenças
print("Carregando modelo BERT...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Modelo carregado com sucesso!")

# Caminhos de entrada e saída
input_path = r"D:\Downloads\ProjTCCMurilo\dataset\BasesTratadas\ptbrData.json"
output_path = r"D:\Downloads\ProjTCCMurilo\correcao_bert_PTBR.json"

def compute_similarity(reference, answer):
    """Calcula a similaridade de cosseno entre a resposta de referência e a resposta do aluno."""
    ref_vec = model.encode(reference).reshape(1, -1)
    ans_vec = model.encode(answer).reshape(1, -1)
    similarity = cosine_similarity(ref_vec, ans_vec)[0][0]
    return similarity

def normalize_grade(similarity, min_sim=0.0, max_sim=1.0, min_score=0.0, max_score=5.0):
    """Mapeia a similaridade de cosseno para uma nota no intervalo desejado (0 a 5)."""
    return round(min_score + (max_score - min_score) * ((similarity - min_sim) / (max_sim - min_sim)), 2)

# Carregar dados JSON
with open(input_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print(f"Processando {len(data)} questões...")

# Processar respostas
for question in data:
    reference_response = question["reference_responses"][0]["reference_response"]
    
    for student_response in question["responses_students"]:
        answer = student_response["answer_question"]
        
        if not answer.strip() or not reference_response.strip():
            similarity = 0.0
        else:
            similarity = compute_similarity(reference_response, answer)
        
        bert_grade = normalize_grade(similarity)
        student_response["bert_similarity"] = float(similarity)
        student_response["bert_grade"] = float(bert_grade)

# Salvar resultados
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(data, output_file, indent=4, ensure_ascii=False)

print(f"Correção concluída! As respostas foram salvas em '{output_path}'")
