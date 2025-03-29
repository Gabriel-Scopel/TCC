import json

# Lista de arquivos para cada embedding
arquivos_elmo = [
    
    "D:/Downloads/ProjTCCMurilo/correcao_elmo_ES.json"
]

arquivos_use = [
    "D:/Downloads/ProjTCCMurilo/correcao_use_ES.json"
]

arquivos_bert = [
    "D:/Downloads/ProjTCCMurilo/correcao_bertES.json"
]

dados_unificados = {}

# Função para carregar e adicionar os dados de ELMO e USE (estrutura normal)
def carregar_e_adicionar(arquivos, chave_nota):
    for arquivo in arquivos:
        with open(arquivo, "r", encoding="utf-8") as f:
            dados_json = json.load(f)
            for item in dados_json:
                # Tenta obter a nota original (original_grade) ou, se não existir, tenta 'grade'
                original_grade = item.get("original_grade") or item.get("grade")
                chave = (item["number_question"], item["answer_question"])  # Criar chave única
                if chave in dados_unificados:
                    dados_unificados[chave][chave_nota] = item[chave_nota]
                    if "original_grade" not in dados_unificados[chave] and original_grade is not None:
                        dados_unificados[chave]["original_grade"] = original_grade
                else:
                    dados_unificados[chave] = {
                        "number_question": item["number_question"],
                        "answer_question": item["answer_question"],
                        chave_nota: item[chave_nota]
                    }
                    if original_grade is not None:
                        dados_unificados[chave]["original_grade"] = original_grade

# Função para carregar e adicionar os dados de BERT (estrutura diferente)
def carregar_e_adicionar_bert(arquivos, chave_nota):
    for arquivo in arquivos:
        with open(arquivo, "r", encoding="utf-8") as f:
            dados_json = json.load(f)
            for item in dados_json:
                for resposta_aluno in item["responses_students"]:  # Acessando dentro de responses_students
                    original_grade = resposta_aluno.get("original_grade") or resposta_aluno.get("grade")
                    chave = (item["number_question"], resposta_aluno["answer_question"])
                    if chave in dados_unificados:
                        dados_unificados[chave][chave_nota] = resposta_aluno[chave_nota]
                        if "original_grade" not in dados_unificados[chave] and original_grade is not None:
                            dados_unificados[chave]["original_grade"] = original_grade
                    else:
                        dados_unificados[chave] = {
                            "number_question": item["number_question"],
                            "answer_question": resposta_aluno["answer_question"],
                            chave_nota: resposta_aluno[chave_nota]
                        }
                        if original_grade is not None:
                            dados_unificados[chave]["original_grade"] = original_grade



# Carregar os dados de cada embedding
carregar_e_adicionar(arquivos_elmo, "elmo_grade")
carregar_e_adicionar(arquivos_use, "use_grade")
carregar_e_adicionar_bert(arquivos_bert, "bert_grade")

# Salvar os dados unificados
output_path = "D:/Downloads/ProjTCCMurilo/dadosES.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(list(dados_unificados.values()), f, ensure_ascii=False, indent=4)

print(f"Dados unificados salvos em {output_path}")
