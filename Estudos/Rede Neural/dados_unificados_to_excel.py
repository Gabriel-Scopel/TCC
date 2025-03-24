import json
import pandas as pd

# Caminho do arquivo JSON
json_file = "dados_unificados.json"

# Carregar os dados do JSON
with open(json_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Criar um DataFrame do Pandas
df = pd.DataFrame(data)

# Garantir que todas as colunas existam, preenchendo com None se necessário
columns = ["number_question", "answer_question", "elmo_grade", "original_grade", "use_grade", "bert_grade"]
df = df.reindex(columns=columns)

# Caminho do arquivo de saída
excel_file = "resultado.xlsx"

# Salvar como arquivo Excel
df.to_excel(excel_file, index=False)

print(f"Arquivo Excel salvo em: {excel_file}")
