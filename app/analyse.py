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
import urllib.request
import json
import geopandas as gpd
import pydeck as pdk
import duckdb

st.title("Analyse du marché immobilier", text_alignment = 'center')
st.markdown("## Tableau de bord", text_alignment = 'center')

#Bloquer la taille de la sidebar
st.markdown("""
<style>
[data-testid="stSidebar"] {
    width: 320px;
    min-width: 320px;
    max-width: 320px;
}
</style>
""", unsafe_allow_html=True)

#--------------------------------------
#----FILTRES GLOBAUX-------------------
#--------------------------------------


def get_min_max_years():
    connexion = duckdb.connect()
    requete = "SELECT MIN(year), MAX(year) FROM 'dvf_clean_2021_2025_v2.parquet'"
    resultat = connexion.execute(requete).fetchone() # fetchone() récupère la première ligne de résultat
    connexion.close()
    
    return int(resultat[0]), int(resultat[1])

min_year, max_year = get_min_max_years()

with st.sidebar:

    st.markdown("### Filtres globaux")

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

        

    st.write("#### Filtrer par DPE :")
    
    dpe_choisis = st.pills(
        label="Choisir les DPE",
        options=["A", "B", "C", "D", "E", "F", "G"],
        selection_mode="multi", 
        default=["A", "B", "C", "D", "E", "F", "G"], 
        label_visibility="collapsed"
    )

if not dpe_choisis:
    st.warning("⚠️ Veuillez sélectionner au moins un DPE dans la barre latérale.")
    st.stop()

nombre_annee_selectionne = selected_year_range[1] - selected_year_range[0] + 1
texte_periode = f"{nombre_annee_selectionne} ans" if nombre_annee_selectionne > 1 else '1 an'

#---Clause paramétré pour la sécurité------
def build_where_clause(year_min, year_max, types_list, dpe_list):
    """
    Construit une clause WHERE sécurisée en utilisant des paramètres liés.
    Retourne la clause et les paramètres.
    """
    conditions = []
    params = []
    
    # Condition années
    conditions.append("year BETWEEN ? AND ?")
    params.extend([year_min, year_max])
    
    # Condition types
    if types_list:
        placeholders = ",".join(["?"] * len(types_list))
        conditions.append(f"\"Type local\" IN ({placeholders})")
        params.extend(types_list)
    
    # Condition DPE
    if dpe_list:
        placeholders = ",".join(["?"] * len(dpe_list))
        conditions.append(f"classe_bilan_dpe IN ({placeholders})")
        params.extend(dpe_list)
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where_clause, params

clause_where_globale, params_globaux = build_where_clause(
    selected_year_range[0],
    selected_year_range[1],
    types_to_filter,
    dpe_choisis
)

#-----------------------
#----VERSION------------
#-----------------------

with st.sidebar:
    st.divider()
    st.caption('v1.1.0', help ="""
    Date de mise à jour : 12/08/26
    
    Nouveautés:
    - Nouvelle page : "Mon projet immobilier"
    - Nouveau filtre : DPE
    - Possibilité de rechercher des informations par adresse
    - Amélioration des performances (passage à DuckDB)
    """,)
    
    st.caption('Made with ❤️ by Thomas')
    st.link_button('Voir le code sur GitHub 👾', "https://github.com/Math1720182/Mon-Analyse-Immo-App")

    
#---------------------------------------------------
#-------Introduction à l'analyse--------------------
#---------------------------------------------------

#--------------------------------------
#----Heatmap intéractive (dept)--------
#--------------------------------------

@st.cache_data
def get_geojson():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else None

@st.cache_data
def get_map_data(clause_where, params):
    requete_sql = f"""
        SELECT 
            "Code departement", 
            "year", 
            "Nom departement", 
            MEDIAN("Price per surface") AS "Price per surface"
        FROM 'dvf_clean_2021_2025_v2.parquet'
        {clause_where}
        GROUP BY "Code departement", "year", "Nom departement"
        ORDER BY "Code departement", "year"
    """
    
    connexion = duckdb.connect()
    df_grouped = connexion.execute(requete_sql, params).df()
    connexion.close()
    
    return df_grouped

geojson_france = get_geojson()
df_map = get_map_data(clause_where_globale, params_globaux)

st.divider()

st.header("Carte des Départements")


@st.fragment
def afficher_section_carte(df_map, geojson_france):
    
    col_carte, col_detail = st.columns([3, 1])
    
    with col_carte:
        st.subheader("Prix médian au m² par département", help = "Pour une analyse fiable il faut utiliser la médiane et non la moyenne. La distribution des données ne suivant pas une loi normale, il faut utiliser la médiane. Pour plus d'informations, visitez la page 'Statistiques'.")

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
                zoom=4.5,
                center={"lat": 46.6, "lon": 2.4},
                opacity=0.7,
                height=500,
                hover_name = 'Nom departement'
            )
            
            fig_map.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                clickmode='event+select',
                uirevision=True  
            )
            
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


    
#--------------------------------------
#---Heatmap intéractive (par commune)--
#--------------------------------------

@st.cache_data
def get_geojson_communes():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/communes.geojson"
    gdf = gpd.read_file(url)
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.004, preserve_topology=True)
    gdf['code_dept'] = gdf['code'].apply(lambda x: x[:3] if x.startswith('97') or x.startswith('98') else x[:2])
    return gdf


@st.cache_data
def get_map_data_communes(clause_where, params):
    requete_sql = f"""
        SELECT
            "Nom departement",
            "Code departement",
            "Code commune",
            "year",
            "Commune",
            MEDIAN ("Price per surface") AS "Price per surface"
        FROM 'dvf_clean_2021_2025_v2.parquet'
        {clause_where}
        GROUP BY "Nom departement", "Code departement", "Code commune", "year", "Commune"
    """

    connexion = duckdb.connect()
    df_grouped = connexion.execute(requete_sql, params).df()
    connexion.close()

    return df_grouped


gdf_france_communes = get_geojson_communes()
df_map_global = get_map_data_communes(clause_where_globale, params_globaux)


st.divider()
st.header("Carte des communes par département")

col1, col2 = st.columns([1,3])
with col1:
    liste_departements = sorted(df_map_global['Nom departement'].dropna().unique())
    index_defaut = liste_departements.index('Oise')
    departement_choisi = st.selectbox("Choisissez un département :", liste_departements, index = index_defaut)

df_dept = df_map_global[df_map_global['Nom departement'] == departement_choisi]

if not df_dept.empty:
    code_dept_val = str(df_dept['Code departement'].iloc[0])
    gdf_dept = gdf_france_communes[gdf_france_communes['code_dept'] == code_dept_val]
else:
    gdf_dept = gpd.GeoDataFrame()

    

@st.fragment
def afficher_section_carte_communes(df_dept_local, gdf_dept_local):
    col_carte, col_detail = st.columns([3, 1])
    
    with col_carte:
        st.subheader("Prix médian au m² par commune")

        if not df_dept_local.empty:

            derniere_annee = df_dept_local['year'].max()
            df_carte_unique = df_dept_local[df_dept_local['year'] == derniere_annee]
            

            vmin = df_carte_unique["Price per surface"].quantile(0.05)
            vmax = df_carte_unique["Price per surface"].quantile(0.95)
            
            if not gdf_dept_local.empty:
                minx, miny, maxx, maxy = gdf_dept_local.total_bounds
                center_lat = (miny + maxy) / 2
                center_lon = (minx + maxx) / 2
                
                fig_map = px.choropleth_mapbox(
                    df_carte_unique,
                    geojson=gdf_dept_local, 
                    locations="Code commune", 
                    featureidkey="properties.code",
                    color="Price per surface",
                    color_continuous_scale="Reds",
                    range_color=(vmin, vmax),
                    mapbox_style="carto-positron",
                    zoom=7.9,
                    hover_name = 'Commune',
                    center={"lat": center_lat, "lon": center_lon},
                    opacity=0.7,
                    height=500
                )
                
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                fig_map.update_layout(clickmode='event+select')
        
                event = st.plotly_chart(fig_map, key="carte_communes_contours", on_select="rerun", use_container_width=True)
        
                if event and event.selection.points:
                    point_index = event.selection.points[0]["point_index"]
                    commune_selectionnee = df_carte_unique.iloc[point_index]["Commune"]
                    st.session_state['commune_cliquee'] = commune_selectionnee
            else:
                st.error("Aucune géométrie trouvée pour ce département.")
        else:
            st.warning("Aucune donnée disponible pour ce département.")

    with col_detail:
        st.markdown("### Zoom")
        commune_a_afficher = st.session_state.get('commune_cliquee')
    
        if commune_a_afficher:
            df_filtered_histo = df_map_global[df_map_global["Commune"] == commune_a_afficher].sort_values('year')
            st.info(f"Historique pour : **{commune_a_afficher}**")
            
            fig_line = px.line(
                df_filtered_histo,
                x="year",
                y="Price per surface",
                markers=True,
                template="plotly_white",
                height=300
            )
            fig_line.update_layout(
                title="Évolution Prix m²",
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

            if pd.notna(annee_max) and pd.notna(annee_min):
                nb_annees = int(annee_max - annee_min)
            else:
                st.warning("Pas assez de données dans sur cette commune")
            
            if first_year_median > 0 and nb_annees > 0:
                evolution = ((last_year_median - first_year_median) / first_year_median) * 100
                label_delta = f"{evolution:+.1f} % sur {nb_annees} ans"
            else:
                label_delta = "N/A"
            
            st.metric(
                label="Prix médian de la zone", 
                value=f"{price_per_surface_median_carte:,.0f} €/m²".replace(',', ' '),
                delta=label_delta
            )
        else:
            st.warning("Veuillez sélectionner une commune sur la carte pour accéder à ses données")

afficher_section_carte_communes(df_dept, gdf_dept)


#-----------------------------------------
#-------Metriques par ville---------------
#-----------------------------------------

st.divider()
st.subheader('Analyse par ville')

@st.cache_data
def get_communes(clause_where, params):

    requete_sql = f"""
        SELECT DISTINCT "Commune"
        FROM "dvf_clean_2021_2025_v2.parquet"
        {clause_where} AND "Commune" IS NOT NULL
        ORDER BY "Commune"
    """

    connexion = duckdb.connect()
    df_resultat = connexion.execute(requete_sql, params).df()
    connexion.close()

    return df_resultat['Commune'].tolist()

@st.cache_data
def get_stats_commune(commune_nom, year_min, year_max, types_list, dpe_list):
    """Version sécurisée - N'accepte pas d'injection via le nom"""
    
    requete_sql = f"""
        SELECT 
            COUNT(*) AS nombre_ventes,
            AVG("Valeur fonciere") AS price_mean_value,
            AVG("Price per surface") AS price_mean_surface,
            MEDIAN("Valeur fonciere") AS price_median_value,
            MEDIAN("Price per surface") AS price_median_surface,
            MIN(year) AS annee_min,
            MAX(year) AS annee_max,
            MEDIAN(CASE WHEN year = (SELECT MIN(year) FROM 'dvf_clean_2021_2025_v2.parquet' 
                WHERE "Commune" = ? AND year BETWEEN ? AND ?
                AND "Type local" IN ({','.join(['?'] * len(types_list))})
                AND classe_bilan_dpe IN ({','.join(['?'] * len(dpe_list))})
            ) THEN "Price per surface" END) AS first_year_median,
            MEDIAN(CASE WHEN year = (SELECT MAX(year) FROM 'dvf_clean_2021_2025_v2.parquet' 
                WHERE "Commune" = ? AND year BETWEEN ? AND ?
                AND "Type local" IN ({','.join(['?'] * len(types_list))})
                AND classe_bilan_dpe IN ({','.join(['?'] * len(dpe_list))})
            ) THEN "Price per surface" END) AS last_year_median
        FROM 'dvf_clean_2021_2025_v2.parquet'
        WHERE "Commune" = ?
          AND year BETWEEN ? AND ?
          AND "Type local" IN ({','.join(['?'] * len(types_list))})
          AND classe_bilan_dpe IN ({','.join(['?'] * len(dpe_list))})
    """
    
    params = (
        [commune_nom, year_min, year_max] + types_list + dpe_list +  # Sous-requête MIN
        [commune_nom, year_min, year_max] + types_list + dpe_list +  # Sous-requête MAX
        [commune_nom, year_min, year_max] + types_list + dpe_list    # Requête principale
    )
    
    connexion = duckdb.connect()
    df_stats = connexion.execute(requete_sql, params).df()
    connexion.close()

    return df_stats.iloc[0].to_dict() if not df_stats.empty else {}

commune = get_communes(clause_where_globale, params_globaux)

if "commune_select" not in st.session_state or st.session_state["commune_select"] not in commune:
    st.session_state["commune_select"] = 'Compiègne (60)'

col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])


with col1:
    commune_select = st.selectbox(
        "🔎 Rechercher une ville :",
        options=commune,
        key = "commune_select",
        help="Tapez les premières lettres pour filtrer la liste"
    )

    stats_commune = get_stats_commune(
    commune_select,
    selected_year_range[0],
    selected_year_range[1],
    types_to_filter,
    dpe_choisis)
    
    nombre_vente_commune = int(stats_commune['nombre_ventes'])

    st.caption(f"Nombre de vente sur {texte_periode}: **{nombre_vente_commune}**")

price_mean_value_commune = stats_commune['price_mean_value']
price_mean_surface_commune = stats_commune['price_mean_surface']
price_median_value_commune = stats_commune['price_median_value']
price_median_surface_commune = stats_commune['price_median_surface']

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
    
    annee_max = stats_commune['annee_max']
    annee_min = stats_commune['annee_min']
    first_year_median = stats_commune['first_year_median']
    last_year_median = stats_commune['last_year_median']

    if pd.notna(annee_max) and pd.notna(annee_min):
        nb_annees = int(annee_max - annee_min)
    else:
        nb_annees = 0

    if pd.notna(first_year_median) and pd.notna(last_year_median) and first_year_median > 0 and nb_annees > 0:
        evolution_commune = ((last_year_median - first_year_median) / first_year_median) * 100
        label_delta_commune = f"{evolution_commune:+.1f} % sur {nb_annees} ans"
    else:
        label_delta_commune = "N/A"

    if pd.notna(price_median_surface_commune):
        valeur_texte = f"{price_median_surface_commune:,.0f} €/m²".replace(",", " ")
    else:
        valeur_texte = "N/A"

    st.metric(
        label="Prix / m² médian",
        value=valeur_texte,
        delta=label_delta_commune
    )

st.divider()

#-------------------------------------------------------------------
#-------Calcul de la moyenne et médiane (par vente et par m²)-------
#-------------------------------------------------------------------

@st.cache_data
def global_kpi(clause_where, params):
    requete_sql = f"""
        SELECT 
            AVG("Valeur fonciere") AS price_mean_value,
            AVG("Price per surface") AS price_mean_surface,
            MEDIAN("Valeur fonciere") AS price_median_value,
            MEDIAN("Price per surface") AS price_median_surface,
        FROM 'dvf_clean_2021_2025_v2.parquet'
        {clause_where}
    """
    connexion = duckdb.connect()
    global_kpi = connexion.execute(requete_sql, params).df()
    connexion.close()

    return global_kpi.iloc[0].to_dict()

global_kpi_return = global_kpi(clause_where_globale, params_globaux)
    
price_mean_value = global_kpi_return['price_mean_value']
price_mean_surface = global_kpi_return['price_mean_surface']
price_median_value = global_kpi_return['price_median_value']
price_median_surface = global_kpi_return['price_median_surface']


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

st.subheader('Analyse par département', help = "Données disponible en téléchargement via l'outil du tableau")

@st.cache_data
def global_kpi_frame(clause_where, params):
    requete_sql = f"""
        SELECT
            "Nom departement" AS nom_departement,
            COUNT(*) AS nombre_vente,
            AVG("Valeur fonciere") AS price_mean_value,
            AVG("Price per surface") AS price_mean_surface,
            MEDIAN("Valeur fonciere") AS price_median_value,
            MEDIAN("Price per surface") AS price_median_surface
        FROM 'dvf_clean_2021_2025_v2.parquet'
        {clause_where}
        GROUP BY "Nom departement"
        ORDER BY "Nom departement"
    """
    connexion = duckdb.connect()
    global_kpi_frame = connexion.execute(requete_sql, params).df()
    connexion.close()

    return global_kpi_frame

global_kpi_frame_return = global_kpi_frame(clause_where_globale, params_globaux)


st.dataframe(
    global_kpi_frame_return.style.format({
        'price_median_value': '{:,.0f} €',
        'price_mean_value': '{:,.0f} €',
        'price_median_surface': '{:,.0f} €/m²',
        'price_mean_surface': '{:,.0f} €/m²',
        'nombre_vente': '{:,}'
    }), hide_index = True
)

st.write("")

st.divider()

#-------------------------------
#----Graph intéractif-----------
#-------------------------------

@st.cache_data
def get_stats_departement(clause_where, params):
    requete_sql = f"""
        SELECT
            "Nom departement",
            COUNT(*) AS nombre_vente,
            AVG("Valeur fonciere") AS prix_moyen,
            MEDIAN("Valeur fonciere") AS prix_median,
            MEDIAN("Price per surface") AS prix_m2_median,

            --Calcul du ratio maison/terrain

            AVG(
                CASE
                    WHEN "Type local" = 'Maison' AND ("Surface terrain" + "Surface reelle bati") > 0
                    THEN ("Surface reelle bati" * 100.0) / ("Surface terrain" + "Surface reelle bati")
                    ELSE NULL
                END
            ) AS maison_sur_terrain

        FROM 'dvf_clean_2021_2025_v2.parquet'
        {clause_where}
        GROUP BY "Nom departement"
    """
    connexion = duckdb.connect()
    df_stats = connexion.execute(requete_sql, params).df()
    connexion.close()

    return df_stats

stats_dep = get_stats_departement(clause_where_globale, params_globaux)

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
def cache_vente_total():
    requete_sql = f"""
        WITH ventes_par_annee AS (
            -- On compte le nombre de ventes par année
            SELECT 
                "year" AS year,
                COUNT(*) AS total_sales
            FROM 'dvf_clean_2021_2025_v2.parquet'
            GROUP BY year
        )
        -- On calcule la variation en % par rapport à l'année précédente
        SELECT 
            year,
            total_sales AS "total sales",
            (total_sales - LAG(total_sales) OVER (ORDER BY year)) * 100.0 
                / LAG(total_sales) OVER (ORDER BY year) AS pct_change
        FROM ventes_par_annee
        WHERE year IS NOT NULL
        ORDER BY year ASC
    """
    connexion = duckdb.connect()
    vente_total = connexion.execute(requete_sql).df()
    connexion.close()

    return vente_total

vente_total = cache_vente_total()

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
def cache_top10_m2(clause_where, params):

    requete_sql = f"""
            SELECT
                "Nom departement",
                "Price per surface"
            FROM 'dvf_clean_2021_2025_v2.parquet'
            {clause_where} AND "Nom departement" IN (
                SELECT "Nom departement"
                FROM 'dvf_clean_2021_2025_v2.parquet'
                {clause_where}
                GROUP BY "Nom departement"
                ORDER BY AVG("Price per surface") DESC
                LIMIT 10
            )
        """

    params_doubles = params + params
    
    connexion = duckdb.connect()
    df_top10 = connexion.execute(requete_sql, params_doubles).df()
    connexion.close()

    top10_m2 = df_top10.groupby('Nom departement')['Price per surface'].mean().sort_values(ascending=False).index.tolist()

    return df_top10, top10_m2

    
df_top10, top10_m2 = cache_top10_m2(clause_where_globale, params_globaux)



def graphique_boxplot(df_top10, top10_m2):

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=None)

    sns.boxplot(
        data = df_top10,
        x='Nom departement',
        y='Price per surface',
        order=top10_m2,
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

    





