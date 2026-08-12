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
import os
import duckdb

st.set_page_config(page_title="Analyse DVF", layout="wide", page_icon = '📈', menu_items = {'About': "Application à but éducatif/académique uniquement. L'ensemble des filtres ne sont que purement personnel. Vous pouvez trouver les statistiques officiel via le portail data.gouv.fr, rubrique 'DVF statistique'. Pour en savoir plus, consultez la page d'introduction"})



NOM_FICHIER_LOCAL = "dvf_clean_2021_2025_v2.parquet"
URL_GITHUB = "https://github.com/Math1720182/dvf_2021_2025_analysis/releases/download/v2.0.0/dvf_clean_2021_2025_v2.parquet"

def telecharger_fichier_si_absent():
    
    if not os.path.exists(NOM_FICHIER_LOCAL):
        with st.spinner("Téléchargement initial de la base DVF en cours..."):
            reponse = requests.get(URL_GITHUB)
            
            with open(NOM_FICHIER_LOCAL, "wb") as fichier:
                fichier.write(reponse.content)
                
            st.success("Téléchargement terminé !")

telecharger_fichier_si_absent()

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

if 'geojson_france' not in st.session_state:
    st.session_state.geojson_france = load_geojson()


if 'toast_notified' not in st.session_state:
    st.toast("Données DVF et carte chargées avec succès !", icon="✅")
    st.session_state.toast_notified = True


pages = {
    "**MENU**": [
        st.Page("intro.py", title="Introduction"),
        st.Page("analyse.py", title="Tableau de bord", default = True),
        st.Page("simulateur.py", title="Mon projet immobilier"),
        st.Page("stats.py", title="Statistiques"),
    ]
}

pg = st.navigation(pages)
pg.run()



