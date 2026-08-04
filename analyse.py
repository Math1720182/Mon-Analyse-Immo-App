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

st.title("Tableau de bord", text_alignment = 'center')
st.markdown("## Analyse du marché immobilier", text_alignment = 'center')

df_clean = st.session_state.df_clean

#--------------------------------------
#----FILTRES GLOBAUX-------------------
#--------------------------------------


with st.sidebar:

    st.markdown("### Filtres globaux")
    
    min_year = int(st.session_state.df_clean['year'].min())
    max_year = int(st.session_state.df_clean['year'].max())

    selected_year_range = st.slider(
        "Période",
        min_value = min_year,
        max_value = max_year,
        value = (min_year, max_year))
    
    type_selection = st.radio(
        "Type de bien",
        options = ['Tous', 'Maison', 'Appartement'],
        horizontal = True)

    if type_selection == 'Tous':
        types_to_filter = ['Maison', 'Appartement']
    else:
        types_to_filter = [type_selection]

df_filtered = st.session_state.df_clean[
    (st.session_state.df_clean['year'].between(selected_year_range[0], selected_year_range[1])) &
    (st.session_state.df_clean['Type local'].isin(types_to_filter))]

nombre_annee_selectionne = df_filtered['year'].nunique() 
texte_periode = f"{nombre_annee_selectionne} ans" if nombre_annee_selectionne > 1 else '1 an'

if df_filtered.empty:
    st.warning("⚠️ Veuillez sélectionner au moins une année et un type de bien dans la barre latérale.")
    st.stop()


#---------------------------------------------------
#-------Introduction à l'analyse--------------------
#---------------------------------------------------

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
    df_grouped = (
        df.groupby(['Code departement', 'year', 'Nom departement'], observed=True)['Price per surface']
        .median()
        .reset_index()
    )
    return df_grouped.sort_values(by=['Code departement', 'year'])

geojson_france = get_geojson()
df_map = get_map_data(df_filtered)

st.divider()

st.header("Carte des Départements")


@st.fragment
def afficher_section_carte(df_map, geojson_france):
    
    col_carte, col_detail = st.columns([3, 1])
    
    with col_carte:
        st.subheader("Prix médian au m² par département")

        #Pour les couleur
        vmin = df_map["Price per surface"].quantile(0.05)
        vmax = df_map["Price per surface"].quantile(0.95)
        
        if geojson_france:
            fig_map = px.choropleth_mapbox(
                df_map,
                geojson=geojson_france,
                locations="Code departement", 
                featureidkey="properties.code",
                color="Price per surface",
                color_continuous_scale="Reds",
                range_color=(vmin,vmax),
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
    
        else:
            st.error("Impossible d'afficher la carte.")


    with col_detail:
        st.markdown("### Zoom")
        
        dep_a_afficher = st.session_state.get('dep_clique')
    
        if dep_a_afficher:
    
            df_filtered_histo = df_map[df_map["Nom departement"] == dep_a_afficher].sort_values('year')
    
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
            
            annee_max = df_filtered_histo['year'].max()
            annee_min = df_filtered_histo['year'].min()

            price_per_surface_median_carte = df_filtered_histo['Price per surface'].median()
            
            last_year_median = df_filtered_histo[df_filtered_histo['year'] == annee_max]['Price per surface'].median()
            first_year_median = df_filtered_histo[df_filtered_histo['year'] == annee_min]['Price per surface'].median()
            
            nb_annees = int(annee_max - annee_min)
            
            if first_year_median > 0 and nb_annees > 0:
                evolution = ((last_year_median - first_year_median) / first_year_median) * 100
                label_delta = f"{evolution:+.1f} % sur {nb_annees} ans"
            else:
                label_delta = "N/A"
            
            st.metric(
                label=f"Prix médian sur {texte_periode}",
                value=f"{price_per_surface_median_carte:,.0f} €/m²",
                delta=label_delta
            )
        else:
            st.warning('Veuillez sélectionner un département pour accéder à ses données')

afficher_section_carte(df_map, geojson_france)

#-----------------------------------------
#-------Metriques par ville---------------
#-----------------------------------------

st.divider()
st.subheader('Analyse par ville')


commune = sorted(df_filtered['Commune'].dropna().unique())

index_defaut = commune.index('COMPIEGNE (60)') if 'COMPIEGNE (60)' in commune else 0

col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])

with col1:
    commune_select = st.selectbox(
        "🔎 Rechercher une ville :",
        options=commune,
        index=index_defaut,
        help="Tapez les premières lettres pour filtrer la liste"
    )
    nombre_vente_commune = df_filtered[df_filtered['Commune']==commune_select]['Price per surface'].count()

    st.caption(f"Nombre de vente sur {texte_periode}: **{nombre_vente_commune}**")

df_commune = df_filtered[df_filtered['Commune']==commune_select]
price_mean_value_commune = df_commune['Valeur fonciere'].mean()
price_mean_surface_commune = df_commune['Price per surface'].mean()
price_median_value_commune = df_commune['Valeur fonciere'].median()
price_median_surface_commune = df_commune['Price per surface'].median()

with col2:
 
    st.metric(
        label="Prix de vente moyen",
        value=f"{price_mean_value_commune:,.0f} €".replace(",", " "),
    )
with col3:
    
    st.metric(
        label="Prix / m² moyen",
        value=f"{price_mean_surface_commune:,.0f} €/m²".replace(",", " "),
    )
with col4:
    
    st.metric(
        label="Prix de vente médian",
        value=f"{price_median_value_commune:,.0f} €".replace(",", " ")
    )
with col5:

    annee_max = df_commune['year'].max()
    annee_min = df_commune['year'].min()
    
    if pd.notna(annee_max) and pd.notna(annee_min):
        nb_annees = int(annee_max - annee_min)
        last_year_median = df_commune[df_commune['year'] == annee_max]['Price per surface'].median()
        first_year_median = df_commune[df_commune['year'] == annee_min]['Price per surface'].median()
    else:
        nb_annees = 0
        first_year_median = 0
    
    if first_year_median > 0 and nb_annees > 0:
        evolution_commune = ((last_year_median - first_year_median) / first_year_median) * 100
        label_delta_commune = f"{evolution_commune:+.1f} % sur {nb_annees} ans"
    else:
        label_delta_commune = "N/A"
    
    st.metric(
        label="Prix / m² médian",
        value=f"{price_median_surface_commune:,.0f} €/m²".replace(",", " "),
        delta=label_delta_commune
    )

st.divider()

#-------------------------------------------------------------------
#-------Calcul de la moyenne et médiane (par vente et par m²)-------
#-------------------------------------------------------------------


price_mean_value = df_filtered['Valeur fonciere'].mean()
price_mean_surface = df_filtered['Price per surface'].mean()
price_median_value = df_filtered['Valeur fonciere'].median()
price_median_surface = df_filtered['Price per surface'].median()


st.subheader(f'Analyse globale du marché sur {texte_periode}')

col1, col2, col3, col4 = st.columns(4)
    
with col1:
    st.metric(
    label="Prix de vente moyen",
    value=f"{price_mean_value:,.0f}€".replace(",", " ")
)

with col2:
    st.metric(
    label="Prix de vente médian",
    value=f"{price_median_value:,.0f}€".replace(",", " ")
)

with col3:
    st.metric(
    label="Prix au m² moyen",
    value=f"{price_mean_surface:,.0f}€".replace(",", " ")
)

with col4:
    st.metric(
    label="Prix au m² médian",
    value=f"{price_median_surface:,.0f}€".replace(",", " ")
)

st.divider()

#------------------------------------------------
#----Tableau moyenne par région et département---
#------------------------------------------------

st.subheader('Analyse par département')

stats_dep = (
    df_filtered.groupby('Nom departement')
    .agg(
        prix_median=('Valeur fonciere', 'median'),
        prix_moyen=('Valeur fonciere', 'mean'),
        prix_m2_median=('Price per surface', 'median'),
        prix_m2_mean=('Price per surface', 'mean'),
        nombre_vente=('Valeur fonciere', 'count'),
    )
    .reset_index()
)

st.dataframe(
    stats_dep.style.format({
        'prix_median': '{:,.0f} €',
        'prix_moyen': '{:,.0f} €',
        'prix_m2_median': '{:,.0f} €/m²',
        'prix_m2_mean': '{:,.0f} €/m²',
        'nombre_vente': '{:,}'
    }), hide_index = True
)

st.write("")

st.divider()

#-------------------------------
#----Graph intéractif-----------
#-------------------------------

@st.cache_data
def calculer_stats_departements(df_filtered, stats_dep):
    df_maison = df_filtered[df_filtered['Type local'] == 'Maison']
    df_maison['maison_sur_terrain'] = df_maison['Surface reelle bati']*100 / (df_maison['Surface terrain'] + df_maison['Surface reelle bati'])
    moyenne_par_dep = df_maison.groupby('Nom departement', observed=True)['maison_sur_terrain'].mean()

    return stats_dep.merge(moyenne_par_dep, on="Nom departement", how='left')

stats_dep = calculer_stats_departements(df_filtered, stats_dep)

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

        top_10 = stats_dep.head(10)
        choix_utilisateur = st.selectbox("Choisissez la métrique pour l'axe Y:", list(options_y.keys()))
        colonne_y = options_y[choix_utilisateur]
        top_10 = stats_dep.sort_values(by=colonne_y, ascending=False).head(10)
        top_10['Nom departement'] = top_10['Nom departement'].astype(str)
        fig, ax = plt.subplots(figsize=(7, 3.8))
    
        
        sns.barplot(
            data=top_10,
            x="Nom departement",
            y=colonne_y,
            palette="Blues_r", 
            ax=ax,
        )
        
        
        ax.set_ylabel(choix_utilisateur, fontsize=11)
        ax.set_xlabel('')
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

    st.write('')
    st.write('')
    
    fig2 = plt.figure(figsize=(7,3.8))
    bar = plt.bar(vente_total['year'], vente_total['total sales'], width = 0.3)
    plt.xlabel('Année')
    plt.ylabel('Nombre de ventes (en million)')
    
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
    plt.title("Nombre de ventes entre 2021 et 2025")

    st.pyplot(fig2)
    plt.close(fig2)

st.divider()
    
#-------------------
#-----Boxplots------
#-------------------

@st.cache_data
def cache_top10_m2(df_filtered):
    top10_m2 = (
        df_filtered.groupby("Nom departement", observed=True)["Price per surface"]
        .mean()
        .nlargest(10)
        .index
    )
    
    df_top10 = df_filtered[df_filtered["Nom departement"].isin(top10_m2)][["Nom departement", "Price per surface"]]

    return df_top10, top10_m2

    
df_top10, top10_m2 = cache_top10_m2(df_filtered)


def graphique_boxplot(df_top10, _top10_m2): #L'underscore dit à Streamlit : "N'essaie pas de lire ni de verifier cet argument pour décider si tu utilises le cache ou non. Contente-toi de l'ignorer pour le cache et passe-le directement à la fonction."

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=None)

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
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    sns.despine(top=True, right=True)

    plt.tight_layout()

    return fig
    
st.subheader(f'Boxplot des prix au m² sur {texte_periode}')

col1, col2, col3 = st.columns([1, 4, 1])

with col2:

    fig_boxplot = graphique_boxplot(df_top10, top10_m2)
    st.pyplot(fig_boxplot)
    plt.close(fig_boxplot)

st.divider()

st.write("⊕ D'autres simulations et statistiques à venir.")

#----------------------------------------------
#------Simulateur de prix de vente-------------
#----------------------------------------------

with st.sidebar:
    st.divider()
    with st.expander("Simuler la valeur de mon bien"):
        with st.form("form_estimation"):
            commune_select = st.selectbox(
                "🔎 Rechercher votre ville :",
                options=commune,
                index=index_defaut,
                help="Tapez les premières lettres pour filtrer la liste\n\n**Estimation basée sur le prix médian au m² de 2025**",
            )
    
            surface = st.number_input(
                "Surface (m²)", min_value=9, max_value=400, value=60
            )
            type_bien = st.selectbox("Type de bien", ["Appartement", "Maison"])
    
            submitted = st.form_submit_button(
                "Estimer mon bien", use_container_width=True
            )
            
        if submitted:
            df_filtre = df_clean[
                (df_clean["Type local"] == type_bien)
                & (df_clean["Commune"] == commune_select)
                & (df_clean["year"] == 2025)
            ]
    
            prix_m2_local = df_filtre["Price per surface"].median()
    
            if pd.notna(prix_m2_local) and prix_m2_local > 0:
                valeur_estimee = surface * prix_m2_local
                estimation_display = f"{valeur_estimee:,.0f} €".replace(",", " ")
            else:
                estimation_display = "Données insuffisantes"
    
            st.metric(
                label="Estimation **indicative** :", value=estimation_display
            )
    


    
#-----------------------
#----VERSION------------
#-----------------------

with st.sidebar:
    st.divider()
    st.caption('v1.0.1')
    st.caption('Made with ❤️ by Thomas')
    st.link_button('Voir le code sur GitHub 👾', "https://github.com/Math1720182/dvf_2021_2025_analysis")

