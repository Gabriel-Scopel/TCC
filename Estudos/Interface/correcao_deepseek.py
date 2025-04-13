import streamlit as st
import ollama
import re

# Frase preestabelecida
frase_preestabelecida = "o céu é azul todos os dias."

# Função para obter a avaliação de similaridade pelo DeepSeek
def obter_nivel_similaridade(frase_usuario):
    prompt = (
        "Avalie a similaridade entre a frase fornecida e a frase preestabelecida numa escala de 0 a 5, "
        "onde 0 significa nenhuma similaridade e 5 significa total similaridade. "
        "Responda apenas com a nota, no formato: Nota: X "
        "Sem explicações, sem observações, apenas a nota.\n\n"
        f"Frase preestabelecida: \"{frase_preestabelecida}\"\n"
        f"Frase fornecida: \"{frase_usuario}\""
    )

    # Chamada ao modelo DeepSeek
    response = ollama.chat(
        model='deepseek-r1:1.5b',
        messages=[{'role': 'user', 'content': prompt}]
    )

    resposta = response['message']['content'].strip()

    # Extrair apenas o número da nota usando regex
    match = re.search(r'Nota:\s*([\d.,]+)', resposta)
    if match:
        return match.group(1)
    else:
        return "Nota não encontrada"

# Interface do Streamlit
def main():
    st.title("Verificador de Similaridade com DeepSeek 🌤️")

    # Entrada do usuário
    frase_usuario = st.text_input("Digite uma frase para comparar:")

    if frase_usuario:
        # Obter o nível de similaridade
        resultado = obter_nivel_similaridade(frase_usuario)

        # Exibir o resultado
        st.markdown("**Nota de similaridade:**")
        st.write(resultado)

# Rodar o app
if __name__ == "__main__":
    main()
