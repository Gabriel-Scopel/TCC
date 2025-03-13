import json

# Carregar a base de dados
with open("D:\Downloads\ProjTCCMurilo\dataset\BasesTratadas\ptbrData.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Converter as notas
for item in data:
    for resposta in item["responses_students"]:
        resposta["grade"] = round((resposta["grade"] / 3) * 5, 2)  # Convertendo para escala de 0 a 5

# Salvar o novo arquivo JSON com as notas ajustadas
with open("ptbrData_normalized.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Conversão concluída! Novo arquivo salvo como 'ptbrData_0a5.json'.")
