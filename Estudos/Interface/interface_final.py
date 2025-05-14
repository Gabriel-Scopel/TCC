import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.express as px
import ollama
import re

# ===========================================
# Funções para carregar os modelos
# ===========================================

@st.cache_resource
def load_bert_model():
    return SentenceTransformer('neuralmind/bert-base-portuguese-cased')  # Modelo local do HuggingFace (auto-cache)

@st.cache_resource
def load_elmo_model():
    return hub.load(r"C:\Users\Gabriel\.cache\kagglehub\models\google\elmo\tensorFlow1\elmo\3")

@st.cache_resource
def load_use_model():
    return hub.load(r"C:\Users\Gabriel\.cache\kagglehub\models\google\universal-sentence-encoder\tensorFlow1\universal-sentence-encoder\2")

bert_model = load_bert_model()
elmo_model = load_elmo_model()
use_model = load_use_model()

# ===========================================
# Funções de Correção
# ===========================================

def compute_similarity_bert(reference, answer):
    ref_vec = bert_model.encode(reference).reshape(1, -1)
    ans_vec = bert_model.encode(answer).reshape(1, -1)
    return cosine_similarity(ref_vec, ans_vec)[0][0]

def compute_similarity_elmo(reference, answer):
    def get_elmo_embedding(text):
        embeddings = elmo_model.signatures["default"](tf.constant([text]))
        return embeddings["default"].numpy()[0]
    ref_vec = get_elmo_embedding(reference).reshape(1, -1)
    ans_vec = get_elmo_embedding(answer).reshape(1, -1)
    return cosine_similarity(ref_vec, ans_vec)[0][0]

def compute_similarity_use(reference, answer):
    def get_use_embedding(text):
        embeddings = use_model.signatures["default"](tf.constant([text]))
        return embeddings["default"].numpy()[0]
    ref_vec = get_use_embedding(reference).reshape(1, -1)
    ans_vec = get_use_embedding(answer).reshape(1, -1)
    return cosine_similarity(ref_vec, ans_vec)[0][0]

def compute_similarity_deepseek(reference, answer):
    prompt = (
        "Avalie a similaridade entre a frase fornecida e a frase preestabelecida numa escala de 0 a 5, "
        "onde 0 significa nenhuma similaridade e 5 significa total similaridade. "
        "Responda apenas com a nota, no formato: Nota: X "
        "Sem explicações, sem observações, apenas a nota.\n\n"
        f"Frase preestabelecida: \"{reference}\"\n"
        f"Frase fornecida: \"{answer}\""
    )
    response = ollama.chat(
        model='deepseek-r1:1.5b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    resposta = response['message']['content'].strip()
    match = re.search(r'Nota:\s*([\d.,]+)', resposta)
    if match:
        return float(match.group(1).replace(',', '.')) / 5.0
    else:
        return 0.0

def normalize_grade(similarity, min_sim=0.0, max_sim=1.0, min_score=0.0, max_score=5.0):
    return round(min_score + (max_score - min_score) * ((similarity - min_sim) / (max_sim - min_sim)), 2)

def weighted_average(grades, weights):
    return round(np.average(grades, weights=weights), 2)

# ===========================================
# Interface Streamlit
# ===========================================

tab1, tab2 = st.tabs(["Correção Automática", "Resultados"])

# -------------------------------------------
# Aba 1 - Correção Automática
# -------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        image = Image.open('fei.jpg')
        resized_image = image.resize((150, 150))
        st.image(resized_image)
    with col2:
        st.title("Projeto Correções Automáticas")

    st.markdown("Insira suas respostas abaixo para as perguntas propostas e receba a correção automática!")

    questions = [
        ("O que é fotossintese e qual sua importância?",
         "A fotossíntese é o processo em que plantas, algas e algumas bactérias transformam luz solar, água e gás carbônico em glicose e oxigênio. Sua importância está na produção de alimento e na liberação de oxigênio essencial para a vida na Terra.")
    ]

    user_answers = {}
    user_name = st.text_input("Digite seu nome:")

    for idx, (question, reference) in enumerate(questions):
        st.markdown(f"### Pergunta {idx + 1}:")
        st.markdown(f"**{question}**")
        with st.expander("Ver resposta de referência"):
            st.write(reference)

        user_answer = st.text_input("Sua resposta:", key=f"resposta_{idx}")
        user_answers[question] = (reference, user_answer)

    if st.button("Corrigir todas as respostas"):
        if not user_name.strip():
            st.warning("Por favor, insira seu nome antes de corrigir.")
        elif any(not answer.strip() for _, answer in user_answers.values()):
            st.warning("Por favor, responda todas as perguntas antes de corrigir.")
        else:
            st.success("Corrigindo...")

            final_scores = []

            for idx, (question, (reference, user_answer)) in enumerate(user_answers.items()):
                bert_grade = normalize_grade(compute_similarity_bert(reference, user_answer))
                elmo_grade = normalize_grade(compute_similarity_elmo(reference, user_answer))
                use_grade = normalize_grade(compute_similarity_use(reference, user_answer))
                deepseek_grade = normalize_grade(compute_similarity_deepseek(reference, user_answer))

                final_grade = weighted_average(
                    [bert_grade, elmo_grade, use_grade, deepseek_grade],
                    weights=[1, 1, 1, 1]
                )
                final_scores.append(final_grade)

                st.markdown(f"### Resultado - Pergunta {idx + 1}")
                df_scores = pd.DataFrame({
                    "Modelo": ["BERT", "ELMo", "USE", "DeepSeek", "Nota Final"],
                    "Nota": [bert_grade, elmo_grade, use_grade, deepseek_grade, final_grade]
                })

                fig = px.bar(
                    df_scores,
                    x="Nota",
                    y="Modelo",
                    orientation="h",
                    text="Nota",
                    color="Modelo",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title=f"Notas por modelo - Pergunta {idx + 1}"
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(xaxis_range=[0, 5])
                st.plotly_chart(fig, use_container_width=True)

            overall_average = round(sum(final_scores) / len(final_scores), 2)

            st.markdown("## Média Geral Final")
            st.metric("Média Geral", f"{overall_average:.2f}")

            try:
                df_existing = pd.read_excel("Usuario.xlsx")
            except FileNotFoundError:
                df_existing = pd.DataFrame()

            new_row = pd.DataFrame([[user_name] + final_scores + [overall_average]],
                                   columns=["Nome"] + [f"Pergunta {i + 1}" for i in range(len(questions))] + ["Média Geral"])

            df_updated = pd.concat([df_existing, new_row], ignore_index=True)
            df_updated.to_excel("Usuario.xlsx", index=False)

# -------------------------------------------
# Aba 2 - Resultados acumulados
# -------------------------------------------
with tab2:
    st.title("Resultados Acumulados")

    try:
        df_results = pd.read_excel("Usuario.xlsx")

        fig1 = px.bar(
            df_results,
            y="Nome",
            x="Média Geral",
            orientation='h',
            title="Média Geral por Usuário",
            labels={"Média Geral": "Nota", "Nome": "Usuário"},
            color="Nome",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig1)

    except FileNotFoundError:
        st.warning("Ainda não há resultados salvos. Faça uma correção para gerar os dados!")
