for student_response in data[0]['responses_students']:
    resposta = student_response['answer_question']
    nota = obter_nota_resposta(resposta)
    print(f"Resposta do aluno: {resposta}")
    print(f"Nota dada: {nota}")
    print("="*50)