import ollama
import json

# Carregar o arquivo JSON com as respostas dos alunos
with open('graded_responses_ptbr.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Função para enviar a resposta ao modelo DeepSeek e obter uma nota
def obter_nota_resposta(resposta):
    response = ollama.chat(model='deepseek-r1:1.5b', messages=[
        {'role': 'user', 'content': f'Avalie a resposta abaixo com uma nota numérica de 0 a 3. Retorne apenas o número, sem nenhuma explicação ou texto adicional.\n\nResposta: "{resposta}"'}
    ])
    return response['message']['content'].strip()

# Para cada resposta dos alunos, submeter ao modelo e imprimir apenas a nota
for student_response in data[0]['responses_students']:
    resposta = student_response['answer_question']
    nota = obter_nota_resposta(resposta)
    print(nota)
