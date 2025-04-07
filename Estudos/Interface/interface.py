import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.express as px

# ===========================================
# Funções para carregar os modelos
# ===========================================

@st.cache_resource
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_elmo_model():
    return hub.load("https://tfhub.dev/google/elmo/3")

@st.cache_resource
def load_use_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

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
    ref_vec = use_model([reference])[0].numpy().reshape(1, -1)
    ans_vec = use_model([answer])[0].numpy().reshape(1, -1)
    return cosine_similarity(ref_vec, ans_vec)[0][0]

# Função para normalizar nota
def normalize_grade(similarity, min_sim=0.0, max_sim=1.0, min_score=0.0, max_score=5.0):
    return round(min_score + (max_score - min_score) * ((similarity - min_sim) / (max_sim - min_sim)), 2)

# Função para calcular média ponderada
def weighted_average(grades, weights):
    return round(np.average(grades, weights=weights), 2)

# ===========================================
# Interface Streamlit
# ===========================================

# Tabs
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

    # Perguntas e respostas de referência
    questions = [
        ("Qual a diferença entre a célula animal e a célula vegetal?",
         "A célula animal e vegetal apresentam formato diferenciado. A célula animal possui formato irregular, enquanto a célula vegetal apresenta uma forma fixa."),

        ("O corpo humano possui vários tipos de células que se organizam, de acordo com suas especializações e funções, formando os tecidos. Quais são as características do tecido epitelial?",
         "O tecido epitelial é caracterizado por células justapostas, pouca matriz extracelular e reveste superfícies internas e externas do corpo, além de formar glândulas."),

        ("Qual a diferença entre fenótipo e genótipo?",
         "Genótipo é a constituição genética de um indivíduo, enquanto fenótipo são as características observáveis resultantes da interação entre o genótipo e o ambiente."),

        ("O que significa transmissão de caracteres hereditários?",
         "É a passagem de características genéticas dos pais para os filhos através dos genes."),

        ("Quais são as diferenças entre veias e artérias?",
         "As artérias transportam sangue do coração para o corpo sob alta pressão, enquanto as veias trazem o sangue de volta ao coração sob baixa pressão.")
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
            st.success("Correção concluída!")

            final_scores = []

            for idx, (question, (reference, user_answer)) in enumerate(user_answers.items()):
                bert_grade = normalize_grade(compute_similarity_bert(reference, user_answer))
                elmo_grade = normalize_grade(compute_similarity_elmo(reference, user_answer))
                use_grade = normalize_grade(compute_similarity_use(reference, user_answer))

                # Média ponderada (igual peso)
                final_grade = weighted_average([bert_grade, elmo_grade, use_grade], weights=[1, 1, 1])
                final_scores.append(final_grade)

                st.markdown(f"**Pergunta {idx + 1}**")
                st.write(f"BERT: {bert_grade} | ELMo: {elmo_grade} | USE: {use_grade}")
                st.write(f"Nota final da questão: **{final_grade}**")

            # Média geral
            overall_average = round(sum(final_scores) / len(final_scores), 2)
            st.markdown(f"## Média geral final: **{overall_average}**")

            # Salvar no Excel acumulando registros
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

        # Gráfico 1: Usuários x Média Geral
        fig1 = px.bar(
            df_results,
            y="Nome",
            x="Média Geral",
            orientation='h',
            title="Média Geral por Usuário",
            labels={"Média Geral": "Nota", "Nome": "Usuário"},
            color="Nome",  # Aqui adiciona cores diferentes por usuário
            color_discrete_sequence=px.colors.qualitative.Safe  # Paleta de cores segura e variada
        )
        st.plotly_chart(fig1)

        # Gráfico 2: Questões mais acertadas
        question_scores = df_results.drop(columns=["Nome", "Média Geral"]).mean().sort_values(ascending=False)

        fig2 = px.bar(
            x=question_scores.index,
            y=question_scores.values,
            title="Média por Questão",
            labels={"x": "Questão", "y": "Nota Média"},
            color=question_scores.index,  # Aqui adiciona cores diferentes por questão
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig2)

    except FileNotFoundError:
        st.warning("Ainda não há resultados salvos. Faça uma correção para gerar os dados!")

