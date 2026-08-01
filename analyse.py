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

st.title("Tableau de bord - Valeurs Foncières (DVF)")

df_clean = st.session_state.df_clean.copy()

#-------------------------------------------------------------------
#-------Introduction à l'analyse------------------------------------
#-------------------------------------------------------------------

#--------------------------------------
#---------Heatmap intéractive----------
#--------------------------------------

@st.cache_data
def get_geojson():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else None

@st.cache_data
def get_map_data(df):
    df_sub = df[['Code departement', 'Nom departement', 'Price per surface', 'Date mutation']].copy()
    df_sub['year'] = df_sub['Date mutation'].astype(str).str[:4]
    
    df_grouped = df_sub.groupby(['Code departement', 'year', 'Nom departement'])['Price per surface'].median().reset_index()
    return df_grouped.sort_values(by=['Code departement', 'year'])

geojson_france = get_geojson()
df_map = get_map_data(df_clean)


st.title("Carte des Départements")

col_carte, col_detail = st.columns([3, 1])

departement_selectionne = None


@st.fragment
def afficher_section_carte(df_map, geojson_france):
    col_carte, col_detail = st.columns([3, 1])
    
    with col_carte:
        st.subheader("Prix médian au m² par département")
    
        if geojson_france:
            fig_map = px.choropleth_mapbox(
                df_map,
                geojson=geojson_france,
                locations="Code departement", 
                featureidkey="properties.code",
                color="Price per surface",
                color_continuous_scale="Reds",
                range_color=(df_map["Price per surface"].min(), df_map["Price per surface"].max()),
                hover_name="Nom departement",
                hover_data={"Price per surface": True, "Code departement": False},
                mapbox_style="carto-positron", # Style de fond clair
                zoom=5,
                center={"lat": 46.6, "lon": 2.4},
                opacity=0.7,
                height=650,
            )
            
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            fig_map.update_layout(clickmode='event+select')
    
            event = st.plotly_chart(fig_map, key="carte_deps_contours", on_select="rerun")
    
            # Logique de récupération du clic sur une zone géographique
            if event and event.selection.points:
                point_index = event.selection.points[0]["point_index"]
                departement_selectionne = df_map.iloc[point_index]["Nom departement"]
                st.session_state['dep_clique'] = departement_selectionne
            
            dep_affiche = st.session_state.get('dep_clique', "Aucun")
            st.caption(f"Département actuellement analysé : **{dep_affiche}**")
    
        else:
            st.error("Impossible d'afficher la carte.")


    with col_detail:
        st.markdown("### Zoom Départemental")
        
        dep_a_afficher = st.session_state.get('dep_clique')
    
        if dep_a_afficher:
    
            df_filtered_histo = df_map[df_map["Nom departement"] == dep_a_afficher]
    
            st.info(f"Historique pour : **{dep_a_afficher}**")
            
            fig_line = px.line(
                df_filtered_histo,
                x="year",
                y="Price per surface",
                markers=True,
                template="plotly_white",
                height=300
            )
            fig_line.update_layout(
                title=f"Évolution Prix m²",
                xaxis_title="",
                yaxis_title="Prix (€/m²)",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
            dernier_prix = df_filtered_histo.iloc[-1]['Price per surface']
            prix_5ans = df_filtered_histo.iloc[0]['Price per surface']
            evolution = ((dernier_prix - prix_5ans) / prix_5ans) * 100
            
            st.metric(
                label="Prix actuel",
                value=f"{dernier_prix:,.0f} €/m²",
                delta=f"{evolution:,.1f} % sur 5 ans"
            )

afficher_section_carte(df_map, geojson_france)

#-------------------------------------------------------------------
#-------Calcul de la moyenne et médiane (par vente et par m²)-------
#-------------------------------------------------------------------


#-----Calcul de la moyenne-----

#---Prix de vente-----
# 1 - Calcul manuel / 2 - Calcul automatique
price_mean_value_manual = (df_clean['Valeur fonciere'].sum() / df_clean['Valeur fonciere'].count())
price_mean_value = df_clean['Valeur fonciere'].mean()

#---Prix par m²---
price_mean_surface = df_clean['Price per surface'].mean()

#------Calcul de la médiane--------

#---Prix de vente---
# 1 - Calcul manuel / 2 - Calcul automatique
prix_tries = df_clean['Valeur fonciere'].dropna().sort_values().reset_index(drop = True)
index_prix = len(prix_tries)//2

if len(prix_tries) % 2 == 0:
    price_median_value_manual = ((prix_tries.iloc[index_prix - 1]) + (prix_tries.iloc[index_prix])) // 2    
else:    
    price_median_value_manual = prix_tries.iloc[index_prix]

price_median_value = df_clean['Valeur fonciere'].median()

#---Prix au m²---
price_median_surface = df_clean['Price per surface'].median()

st.header('1. Analyse globale du marché entre 2021 et 2025')

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
    label="Prix de vente moyen",
    value=f"{price_mean_value:,.0f}€"
)

with col2:
    st.metric(
    label="Prix de vente médian",
    value=f"{price_median_value:,.0f}€"
)

with col3:
    st.metric(
    label="Prix au m² moyen",
    value=f"{price_mean_surface:,.0f}€"
)

with col4:
    st.metric(
    label="Prix au m² médian",
    value=f"{price_median_surface:,.0f}€"
)

#------------------------------------------------
#-------Moyenne par région et département--------
#------------------------------------------------

st.header('Analyse par département')

stats_dep = (
    df_clean.groupby('Nom departement')
    .agg(
        prix_median=('Valeur fonciere', 'median'),
        prix_moyen=('Valeur fonciere', 'mean'),
        prix_m2_median=('Price per surface', 'median'),
        prix_m2_mean=('Price per surface', 'mean'),
        nombre_vente=('Valeur fonciere', 'count'),
    )
    .reset_index()
)

stats_dep = stats_dep.sort_values(by = 'prix_median', ascending = False)

st.dataframe(
    stats_dep.style.format({
        'prix_median': '{:,.0f} €',
        'prix_moyen': '{:,.0f} €',
        'prix_m²_median': '{:,.0f} €/m²',
        'prix_m²_moyen': '{:,.0f} €/m²',
        'nombre_vente': '{:,}'
    }), hide_index = True
)

st.write("")

#-------------------------------
#----Graph intéractif-----------
#-------------------------------

@st.cache_data
def calculer_stats_departements(df_clean, stats_dep):
    df_maison = df_clean[df_clean['Type local'] == 'Maison'].copy()
    df_maison['maison_sur_terrain'] = df_maison['Surface reelle bati']*100 / (df_maison['Surface terrain'] + df_maison['Surface reelle bati'])
    moyenne_par_dep = df_maison.groupby('Nom departement')['maison_sur_terrain'].mean()

    return stats_dep.merge(moyenne_par_dep, on="Nom departement", how='left')

stats_dep = calculer_stats_departements(df_clean, stats_dep)

#Graphique

col1, col2 = st.columns([2, 2])


with col1:

    @st.fragment 

    def afficher_graphique_dynamique(stats_dep):
        
        options_y = {
        "Prix médian (€)": "prix_median",
        "Prix moyen (€)": "prix_moyen",
        "Prix au m² médian (€/m²)": "prix_m2_median",
        "Nombre de ventes": "nombre_vente",
        "Ratio surface maison sur terrain": "maison_sur_terrain"
        }
        
        choix_utilisateur = st.selectbox("Choisissez la métrique pour l'axe Y:", list(options_y.keys()))
        colonne_y = options_y[choix_utilisateur]
        top_10 = stats_dep.head(10)
        fig, ax = plt.subplots(figsize=(7, 3.8))
    
        
        sns.barplot(
            data=top_10,
            x="Nom departement",
            y=colonne_y,
            palette="Blues_r", 
            ax=ax,
        )
        
        
        ax.set_ylabel(choix_utilisateur, fontsize=11)
        ax.set_title(
            f"Top 10 des départements par {choix_utilisateur}",
            fontsize=12,
            weight="bold",
        )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    afficher_graphique_dynamique(stats_dep)

#----------------------------------
#---------Nombre de vente----------
#----------------------------------

@st.cache_data
def cache_vente_total(df_clean):
    
    vente_total = df_clean['year'].value_counts().reset_index()
    vente_total.columns = ['year', 'total sales']
    vente_total['year'] = pd.to_numeric(vente_total['year'], errors = 'coerce')
    vente_total = vente_total.sort_values(by='year', ascending =True).reset_index(drop=True)
    vente_total['pct_change'] = vente_total['total sales'].pct_change() * 100

    return vente_total
    
vente_total = cache_vente_total(df_clean)

with col2:
    fig2 = plt.figure(figsize=(7,3.8))
    bar = plt.bar(vente_total['year'], vente_total['total sales'], width = 0.3)
    plt.xlabel('Year')
    plt.ylabel('Number of sales in million')
    
    #Ajout des pourcentages d'évolution
    for i, bar in enumerate(bar):
        yval = bar.get_height()
    
        if i > 0:
            pct = vente_total.loc[i, 'pct_change']
            if pct > 0:
                signe = '+'
            else:
                signe = ''
            texte = f'{signe}{pct:.1f}%'
    
            couleur_texte = 'green' if pct>= 0 else 'red'
    
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 30000,
                texte,
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                color=couleur_texte,
            )

    plt.tight_layout()
    plt.title("Number of sales between 2021 and 2025")

    st.pyplot(fig2)
    plt.close(fig2)

st.divider()
    
#-------------------
#-----Boxplots------
#-------------------

@st.cache_data
def cache_top10_m2(df_clean):
    top10_m2 = (
        df_clean.groupby("Nom departement")["Price per surface"]
        .mean()
        .nlargest(10)
        .index
    )
    
    df_top10 = df_clean[df_clean["Nom departement"].isin(top10_m2)]

    return df_top10, top10_m2
    
df_top10, top10_m2 = cache_top10_m2(df_clean)


def graphique_boxplot(df_top10, _top10_m2): #L'underscore dit à Streamlit : "N'essaie pas de lire ni de verifier cet argument pour décider si tu utilises le cache ou non. Contente-toi de l'ignorer pour le cache et passe-le directement à la fonction."

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')

    sns.boxplot(
        data = df_top10,
        x='Nom departement',
        y='Price per surface',
        order=_top10_m2,
        color='#8ecae6', 
        width=0.5,
        showfliers=False,
        ax=ax,
        boxprops=dict(alpha=0.8, edgecolor="#219ebc", linewidth=1.5),
        medianprops=dict(color="#fb8500", linewidth=2),

    )

    ax.set_title('Top 10 des départements au prix du m²', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Prix au m² (€)', fontsize=10, fontweight='bold', labelpad=8)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    sns.despine(top=True, right=True)

    plt.tight_layout()

    return fig
    
st.subheader('Boxplot des prix au m²')

col1, col2, col3 = st.columns([1, 4, 1])

with col2:

    fig_boxplot = graphique_boxplot(df_top10, top10_m2)
    st.pyplot(fig_boxplot)
    plt.close(fig_boxplot)

st.divider()

