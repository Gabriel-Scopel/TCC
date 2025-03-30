import ollama
import json

# Carregar o arquivo JSON com as respostas dos alunos
with open('graded_responses_ptbr.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Função para enviar a resposta ao modelo DeepSeek e obter uma nota
def obter_nota_resposta(resposta):
    response = ollama.chat(model='deepseek-r1:1.5b', messages=[
        {'role': 'user', 'content': f'Por favor, avalie a seguinte resposta em uma escala de 0 a 3: "{resposta}"'}
    ])
    # Acessa a resposta e retorna a nota
    return response['message']['content']

# Para cada resposta dos alunos, submeter ao modelo e imprimir a nota
for student_response in data[0]['responses_students']:
    resposta = student_response['answer_question']
    nota = obter_nota_resposta(resposta)
    print(f"Resposta do aluno: {resposta}")
    print(f"Nota dada: {nota}")
    print("="*50)
