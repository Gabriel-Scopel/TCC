import json
import torch
from sentence_transformers import SentenceTransformer, util

# Carregar modelo BERT pré-treinado para embeddings de sentenças
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(reference, answer):
    """Calcula a similaridade entre a resposta de referência e a resposta do aluno usando BERT."""
    ref_embedding = model.encode(reference, convert_to_tensor=True)
    ans_embedding = model.encode(answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(ref_embedding, ans_embedding).item()
    return similarity

def normalize_grade(similarity, max_score=3.0):
    """Converte a similaridade em uma nota proporcional ao escore máximo."""
    return round(similarity * max_score, 1)  # Normaliza para escala de 0 a 3

# Carregar arquivo JSON com as respostas
with open("ptbrData.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Processar respostas
for question in data:
    reference_response = question["reference_responses"][0]["reference_response"]
    
    for student_response in question["responses_students"]:
        answer = student_response["answer_question"]
        similarity = compute_similarity(reference_response, answer)
        bert_grade = normalize_grade(similarity)
        student_response["bert_grade"] = bert_grade

# Salvar resultados em um novo arquivo JSON
with open("graded_responses_ptbr.json", "w", encoding="utf-8") as output_file:
    json.dump(data, output_file, indent=4, ensure_ascii=False)

print("Correção concluída! As respostas foram salvas em 'graded_responses_ptbr.json'")
