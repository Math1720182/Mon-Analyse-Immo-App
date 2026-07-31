import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.stats import norm, kstest, mannwhitneyu, spearmanr, kruskal
from scipy.stats import t as student_t
import seaborn as sns
import itertools
import streamlit as st
import plotly.express as px
import requests

st.set_page_config(page_title="Analyse DVF", layout="wide", page_icon = '📈', menu_items = {'About': "Application à but éducatif/académique uniquement. L'ensemble des filtres ne sont que purement personnel. Vous pouvez trouver les statistiques officiel via le portail data.gouv.fr, rubrique 'DVF statistique'. Pour en savoir plus, consultez la page d'introduction"})

@st.cache_data
def load_clean_data():
    return pd.read_parquet("https://github.com/Math1720182/dvf_2021_2025_analysis/releases/download/v1.0.0/dvf_clean_2021_2025.parquet")

@st.cache_data 
def load_geojson():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors du chargement du fond de carte")
        return None

# --- Chargement dans Session State ---
if 'df_clean' not in st.session_state:
    with st.spinner("Chargement des données DVF..."):
        st.session_state.df_clean = load_clean_data()

if 'geojson_france' not in st.session_state:
    st.session_state.geojson_france = load_geojson()


if 'toast_notified' not in st.session_state:
    st.toast("Données DVF et carte chargées avec succès !", icon="✅")
    st.session_state.toast_notified = True


intro_page = st.Page("intro.py", title="1. Introduction", default=True)
analyse_page = st.Page("analyse.py", title="2. Analyse")
stats_page = st.Page("stats.py", title="3. Tests statistique")

pg = st.navigation([intro_page, analyse_page, stats_page])
pg.run()



