import ollama
import json
import re
from tqdm import tqdm

# Carregar o arquivo JSON com as respostas dos alunos (enData.json)
with open(r'D:/Downloads/ProjTCCMurilo/dataset/BasesTratadas/esData.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Função para enviar a resposta ao modelo DeepSeek e obter a avaliação
def obter_avaliacao_resposta(resposta):
    mensagem = (
        "Por favor, avalie a seguinte resposta em uma escala de 0 a 5 e retorne também "
        "a similaridade de cosseno em uma escala de 0 a 1. Responda no formato JSON com os campos "
        "'grade', 'cosine_similarity' e 'full_response' contendo todos os detalhes da avaliação. "
        f"A resposta é: \"{resposta}\""
    )
    response = ollama.chat(model='deepseek-r1:1.5b', messages=[{'role': 'user', 'content': mensagem}])
    full_response = response['message']['content']
    
    # Tentar extrair o bloco JSON delimitado por ```json e ```
    match = re.search(r'```json\s*(\{.*\})\s*```', full_response, re.DOTALL)
    if match:
        json_text = match.group(1)
    else:
        # Se não houver delimitadores, tentar extrair o primeiro objeto JSON
        match = re.search(r'(\{.*\})', full_response, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            json_text = None
    
    if json_text:
        try:
            avaliacao = json.loads(json_text)
        except Exception as e:
            avaliacao = {
                "grade": 0,
                "cosine_similarity": 0,
                "full_response": full_response
            }
    else:
        avaliacao = {
            "grade": 0,
            "cosine_similarity": 0,
            "full_response": full_response
        }
    
    return avaliacao

# Listas para armazenar as duas saídas
saida_modelo = []         # Dados completos de todas as avaliações do modelo
resultado_alunos = []     # Estrutura seguindo o layout do correcao_bertEN.json

# Processa cada questão presente no arquivo de entrada
for question in tqdm(data, desc="Processando questões"):
    question_number = question.get('number_question')
    question_text = question.get('question_text')
    keywords = question.get('keywords', [])
    reference_responses = question.get('reference_responses', [])
    
    student_full_details = []  # Saída completa com os dados retornados pelo modelo
    student_structured = []      # Saída estruturada conforme o layout de correcao_bertEN.json
    
    # Processa cada resposta dos alunos para a questão corrente
    for student_response in question.get('responses_students', []):
        resposta = student_response.get('answer_question')
        # Preserva a nota original do enData.json
        orig_grade = student_response.get('grade')
        avaliacao = obter_avaliacao_resposta(resposta)
        
        # Extração dos campos de avaliação do modelo DeepSeek
        dpseek_grade = avaliacao.get('grade', 0)
        dpseek_similarity = avaliacao.get('cosine_similarity', 0)
        
        # Adiciona à saída completa, mantendo o campo 'grade' original
        student_full_details.append({
            'number_question': question_number,
            'answer_question': resposta,
            'grade': orig_grade,
            'avaliacao_modelo': avaliacao  # inclui dpseek_grade, dpseek_similarity e full_response
        })
        
        # Adiciona à saída estruturada seguindo o layout de referência, sem sobrescrever a nota original
        student_structured.append({
            'number_question': question_number,
            'answer_question': resposta,
            'grade': orig_grade,  # nota original do enData.json
            'dpseek_grade': dpseek_grade,  # nota determinada pelo modelo DeepSeek
            'dpseek_similarity': dpseek_similarity  # similaridade de cosseno determinada pelo modelo
        })
    
    question_full = {
         'number_question': question_number,
         'question_text': question_text,
         'keywords': keywords,
         'reference_responses': reference_responses,
         'responses_students': student_full_details
    }
    question_structured = {
         'number_question': question_number,
         'question_text': question_text,
         'keywords': keywords,
         'reference_responses': reference_responses,
         'responses_students': student_structured
    }
    
    saida_modelo.append(question_full)
    resultado_alunos.append(question_structured)


# Salvar a correção dos alunos seguindo o layout do correcao_bertEN.json
with open(r'D:/Downloads/ProjTCCMurilo/Ollama/deepseekES.json', 'w', encoding='utf-8') as f:
    json.dump(resultado_alunos, f, ensure_ascii=False, indent=4)

print("Resultados salvos em 'DeepSeekES.json'.")
