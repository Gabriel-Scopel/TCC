import json

# Lista de arquivos para cada embedding
arquivos_elmo = [
    "D:\Downloads\ProjTCCMurilo\correcoes\correcao_elmo_EN.json",
]

arquivos_use = [
    "D:\Downloads\ProjTCCMurilo\correcoes\correcao_use_EN.json",
]

arquivos_bert = [
    "D:\Downloads\ProjTCCMurilo\correcoes\correcao_bertEN.json",
]

arquivos_deepseek = [
    "D:\Downloads\ProjTCCMurilo\Ollama\deepseekEN.json",
]

dados_unificados = {}

# Função para carregar e adicionar os dados de ELMO e USE (estrutura normal)
def carregar_e_adicionar(arquivos, chave_nota):
    # Define a chave de similaridade correspondente: ex. "elmo_grade" -> "elmo_similarity"
    similarity_key = chave_nota.replace("grade", "similarity")
    for arquivo in arquivos:
        with open(arquivo, "r", encoding="utf-8") as f:
            dados_json = json.load(f)
            for item in dados_json:
                original_grade = item.get("original_grade") or item.get("grade")
                chave = (item["number_question"], item["answer_question"])  # chave única
                if chave in dados_unificados:
                    dados_unificados[chave][chave_nota] = item[chave_nota]
                    if similarity_key in item:
                        dados_unificados[chave][similarity_key] = item[similarity_key]
                    if "original_grade" not in dados_unificados[chave] and original_grade is not None:
                        dados_unificados[chave]["original_grade"] = original_grade
                else:
                    dados_unificados[chave] = {
                        "number_question": item["number_question"],
                        "answer_question": item["answer_question"],
                        chave_nota: item[chave_nota]
                    }
                    if similarity_key in item:
                        dados_unificados[chave][similarity_key] = item[similarity_key]
                    if original_grade is not None:
                        dados_unificados[chave]["original_grade"] = original_grade

# Função para carregar e adicionar os dados de BERT (estrutura diferente)
def carregar_e_adicionar_bert(arquivos, chave_nota):
    similarity_key = chave_nota.replace("grade", "similarity")
    for arquivo in arquivos:
        with open(arquivo, "r", encoding="utf-8") as f:
            dados_json = json.load(f)
            for item in dados_json:
                for resposta_aluno in item["responses_students"]:
                    original_grade = resposta_aluno.get("original_grade") or resposta_aluno.get("grade")
                    chave = (item["number_question"], resposta_aluno["answer_question"])
                    if chave in dados_unificados:
                        dados_unificados[chave][chave_nota] = resposta_aluno[chave_nota]
                        if similarity_key in resposta_aluno:
                            dados_unificados[chave][similarity_key] = resposta_aluno[similarity_key]
                        if "original_grade" not in dados_unificados[chave] and original_grade is not None:
                            dados_unificados[chave]["original_grade"] = original_grade
                    else:
                        dados_unificados[chave] = {
                            "number_question": item["number_question"],
                            "answer_question": resposta_aluno["answer_question"],
                            chave_nota: resposta_aluno[chave_nota]
                        }
                        if similarity_key in resposta_aluno:
                            dados_unificados[chave][similarity_key] = resposta_aluno[similarity_key]
                        if original_grade is not None:
                            dados_unificados[chave]["original_grade"] = original_grade

# Função para carregar e adicionar os dados de DEEPSEEK (estrutura semelhante à BERT)
def carregar_e_adicionar_deepseek(arquivos, chave_nota):
    similarity_key = chave_nota.replace("grade", "similarity")
    for arquivo in arquivos:
        with open(arquivo, "r", encoding="utf-8") as f:
            dados_json = json.load(f)
            for item in dados_json:
                for resposta_aluno in item["responses_students"]:
                    # Utiliza os campos dpseek_grade e dpseek_similarity
                    original_grade = resposta_aluno.get("original_grade") or resposta_aluno.get("grade")
                    chave = (item["number_question"], resposta_aluno["answer_question"])
                    if chave in dados_unificados:
                        dados_unificados[chave][chave_nota] = resposta_aluno.get(chave_nota)
                        if similarity_key in resposta_aluno:
                            dados_unificados[chave][similarity_key] = resposta_aluno[similarity_key]
                    else:
                        dados_unificados[chave] = {
                            "number_question": item["number_question"],
                            "answer_question": resposta_aluno["answer_question"],
                            chave_nota: resposta_aluno.get(chave_nota)
                        }
                        if similarity_key in resposta_aluno:
                            dados_unificados[chave][similarity_key] = resposta_aluno[similarity_key]
                        if original_grade is not None:
                            dados_unificados[chave]["original_grade"] = original_grade

# Carregar os dados de cada embedding
carregar_e_adicionar(arquivos_elmo, "elmo_grade")
carregar_e_adicionar(arquivos_use, "use_grade")
carregar_e_adicionar_bert(arquivos_bert, "bert_grade")
carregar_e_adicionar_deepseek(arquivos_deepseek, "dpseek_grade")

# Salvar os dados unificados
output_path = "D:/Downloads/ProjTCCMurilo/dadosEN.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(list(dados_unificados.values()), f, ensure_ascii=False, indent=4)

print(f"Dados unificados salvos em {output_path}")
