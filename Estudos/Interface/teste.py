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
    return SentenceTransformer('neuralmind/bert-base-portuguese-cased')

@st.cache_resource
def load_elmo_model():
    return hub.load("https://tfhub.dev/google/elmo/3")  # Versão estável do ELMo

@st.cache_resource
def load_use_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")  # USE mais recente

bert_model = load_bert_model()
elmo_model = load_elmo_model()
use_model = load_use_model()

# ===========================================
# Funções de extração de embeddings
# ===========================================

def get_embedding_elmo(text):
    """Extrai embedding usando o modelo ELMo."""
    embeddings = elmo_model.signatures["default"](tf.constant([text]))
    return embeddings["default"].numpy()[0]

def get_embedding_deepseek(text):
    """
    Extrai um vetor de embedding via DeepSeek.
    O prompt deve instruir o modelo a responder apenas com uma lista de números (separados por vírgula).
    """
    prompt = (
        "Extraia um vetor de embedding para o seguinte texto. "
        "Responda apenas com uma lista de números, separados por vírgula, representando o vetor.\n\n"
        f"Texto: \"{text}\""
    )
    response = ollama.chat(
        model='deepseek-r1:1.5b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    content = response['message']['content'].strip()
    try:
        # Procura números (incluindo decimais) e converte para float
        numbers = [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", content)]
        vector = np.array(numbers)
        return vector
    except Exception as e:
        return None

# ===========================================
# Função auxiliar para calcular similaridade de cosseno de forma segura
# ===========================================

def safe_cosine_similarity(vec1, vec2):
    """
    Calcula a similaridade de cosseno entre dois vetores,
    preenchendo (pad) com zeros o vetor de dimensão menor, se necessário.
    """
    if vec1.shape != vec2.shape:
        # Determina a dimensão máxima
        max_dim = max(vec1.shape[0], vec2.shape[0])
        # Preenche o vetor menor com zeros até alcançar a dimensão do maior
        if vec1.shape[0] < max_dim:
            pad_width = max_dim - vec1.shape[0]
            vec1 = np.pad(vec1, (0, pad_width), 'constant')
        if vec2.shape[0] < max_dim:
            pad_width = max_dim - vec2.shape[0]
            vec2 = np.pad(vec2, (0, pad_width), 'constant')
    return cosine_similarity([vec1], [vec2])[0][0]

# ===========================================
# Função de Similaridade e Cálculo de Nota
# ===========================================

def compute_all_similarities(reference, answer):
    """
    Para cada técnica, extrai o vetor da resposta gabarito e da resposta do visitante,
    calcula a similaridade de cosseno e retorna os valores.
    """
    # BERT
    emb_bert_ref = bert_model.encode(reference)
    emb_bert_ans = bert_model.encode(answer)
    sim_bert = cosine_similarity([emb_bert_ref], [emb_bert_ans])[0][0]
    
    # ELMo
    emb_elmo_ref = get_embedding_elmo(reference)
    emb_elmo_ans = get_embedding_elmo(answer)
    sim_elmo = cosine_similarity([emb_elmo_ref], [emb_elmo_ans])[0][0]
    
    # Universal Sentence Encoder
    emb_use_ref = use_model([reference])[0].numpy()
    emb_use_ans = use_model([answer])[0].numpy()
    sim_use = cosine_similarity([emb_use_ref], [emb_use_ans])[0][0]
    
    # DeepSeek
    emb_deepseek_ref = get_embedding_deepseek(reference)
    emb_deepseek_ans = get_embedding_deepseek(answer)
    if emb_deepseek_ref is None or emb_deepseek_ans is None:
        sim_deepseek = 0.0
    else:
        sim_deepseek = safe_cosine_similarity(emb_deepseek_ref, emb_deepseek_ans)
    
    return sim_bert, sim_elmo, sim_use, sim_deepseek

def normalize_grade(similarity, min_sim=0.0, max_sim=1.0, min_score=0.0, max_score=5.0):
    """
    Normaliza uma similaridade (esperada entre 0 e 1) para uma nota na escala de 0 a 5.
    """
    return round(min_score + (max_score - min_score) * ((similarity - min_sim) / (max_sim - min_sim)), 2)

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
    
    # Pergunta e resposta de referência (gabarito)
    questions = [
        ("O que é fotossíntese e qual sua importância?",
         "A fotossíntese é o processo em que plantas, algas e algumas bactérias transformam luz solar, água e gás carbônico em glicose e oxigênio. "
         "Sua importância está na produção de alimento e na liberação de oxigênio essencial para a vida na Terra.")
    ]
    
    user_answers = {}
    user_name = st.text_input("Digite seu nome:")
    
    for idx, (question, reference) in enumerate(questions):
        st.markdown(f"### Pergunta {idx + 1}:")
        st.markdown(f"{question}")
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
                # Calcula as similaridades para cada técnica
                sim_bert, sim_elmo, sim_use, sim_deepseek = compute_all_similarities(reference, user_answer)
                
                # Normaliza cada similaridade para uma nota de 0 a 5
                bert_grade = normalize_grade(sim_bert)
                elmo_grade = normalize_grade(sim_elmo)
                use_grade  = normalize_grade(sim_use)
                deepseek_grade = normalize_grade(sim_deepseek)
                
                # Calcula a média ponderada (peso igual para cada técnica)
                final_grade = weighted_average([bert_grade, elmo_grade, use_grade, deepseek_grade],
                                               weights=[1, 1, 1, 1])
                final_scores.append(final_grade)
                
                st.markdown(f"Pergunta {idx + 1}")
                st.write(f"BERT: {bert_grade} | ELMo: {elmo_grade} | USE: {use_grade} | DeepSeek: {deepseek_grade}")
                st.write(f"Nota final da questão: {final_grade}")
            
            overall_average = round(sum(final_scores) / len(final_scores), 2)
            st.markdown(f"## Média geral final: {overall_average}")
            
            # Salvar os resultados em um arquivo Excel
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
        
        question_scores = df_results.drop(columns=["Nome", "Média Geral"]).mean().sort_values(ascending=False)
        fig2 = px.bar(
            x=question_scores.index,
            y=question_scores.values,
            title="Média por Questão",
            labels={"x": "Questão", "y": "Nota Média"},
            color=question_scores.index,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig2)
        
    except FileNotFoundError:
        st.warning("Ainda não há resultados salvos. Faça uma correção para gerar os dados!")