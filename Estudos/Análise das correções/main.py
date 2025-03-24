import json
import pandas as pd

# Carregar os dados do JSON
with open('graded_responses_ptbr.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Criar uma lista para armazenar os dados extraídos
rows = []

# Percorrer os dados e extrair as informações necessárias
for entry in data:
    for response in entry.get("responses_students", []):
        rows.append({
            "number_question": response["number_question"],
            "answer_question": response["answer_question"],
            "grade": response["grade"],
            "bert_grade": response["bert_grade"]
        })

# Criar um DataFrame do pandas
df = pd.DataFrame(rows)

# Salvar o DataFrame em um arquivo Excel
excel_path = "graded_responses.xlsx"
df.to_excel(excel_path, index=False)

print(f"Arquivo Excel salvo em: {excel_path}")
