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
import logging

#---Configuration du logging sécurisé (pas d'exposition de stack trace)----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



st.title("Analyse du marché immobilier", text_alignment = 'center')
st.markdown("## Mon projet immobilier", text_alignment = 'center')
st.write("")

st.markdown("###### 1 - Estimez votre bien (actuel ou futur) sur les dernières ventes dans un rayon proche de chez vous pour une estimation précise !")
st.markdown("###### 2 - Utilisez les informations complémentaires pour analyser l'environnement du bien.")
st.write("*Données issus de sources fiables et officiels (data.gouv.fr, transports.gouv.fr, data.sncf etc.)*")
st.divider()

#-----------------------
#----VERSION------------
#-----------------------

with st.sidebar:

    st.caption('v1.1.0', help ="""
    Date de mise à jour : 12/08/26
    
    Nouveautés:
    - Nouvelle page : "Mon projet immobilier"
    - Nouveau filtre : DPE
    - Possibilité de rechercher des informations par adresse
    - Amélioration des performances (passage à DuckDB)
    """,)
    
    st.caption('Made with ❤️ by Thomas')
    st.link_button('Voir le code sur GitHub 👾', "https://github.com/Math1720182/dvf_2021_2025_analysis")

#-------------------------------
#----Simulateur de bien---------
#-------------------------------

@st.cache_data
def get_zone_sales(code_commune, type_bien, lat_target, lon_target, rayon_m):
    """Version sécurisée avec paramètres liés"""
    # Formule de Haversine directement traduite en SQL
    requete_sql = f"""
        WITH ventes_commune AS (
            SELECT 
                *,
                2 * 6371000 * ASIN(
                    SQRT(
                        POWER(SIN(RADIANS(latitude - ?) / 2.0), 2) +
                        COS(RADIANS(?)) * COS(RADIANS(latitude)) *
                        POWER(SIN(RADIANS(longitude - ?) / 2.0), 2)
                    )
                ) AS distance_m
            FROM 'dvf_clean_2021_2025_v2.parquet'
            WHERE "Code commune" = ?
              AND "Type local" = ?
              AND latitude IS NOT NULL
              AND longitude IS NOT NULL
        )
        SELECT *
        FROM ventes_commune
        WHERE distance_m <= ?
        ORDER BY distance_m ASC
    """
    
    params = [lat_target, lat_target, lon_target, code_commune, type_bien, rayon_m]
    
    connexion = duckdb.connect()
    df_zone = connexion.execute(requete_sql, params).df()
    connexion.close()

    return df_zone
    

def geocode_adresse(adresse):
    try:
        # Utilise requests.utils.quote pour encoder l'URL de manière sûre
        url = f"https://api-adresse.data.gouv.fr/search/?q={requests.utils.quote(adresse)}&limit=1"
        
        res = requests.get(url, timeout=5).json()
        
        if res.get('features'):
            feature = res['features'][0]
            lon, lat = feature['geometry']['coordinates']
            label = feature['properties']['label']
            code_commune = feature['properties']['citycode']
            commune = feature['properties']['city']
            return lat, lon, label, str(code_commune), commune
    
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout lors du géocodage de l'adresse")
        st.error("Délai d'attente dépassé. Veuillez réessayer.")
    except requests.exceptions.ConnectionError:
        logger.warning(f"Erreur de connexion lors du géocodage")
        st.error("Erreur de connexion. Vérifiez votre connexion Internet.")
    except ValueError:
        logger.warning(f"Erreur de parsing JSON lors du géocodage")
        st.error("Réponse invalide du serveur de géocodage.")
    except Exception as e:
        # Log le type d'erreur, pas les détails sensibles
        logger.error(f"Erreur de géocodage: {type(e).__name__}")
        st.error("Adresse introuvable. Précisez le numéro, la rue et la ville.")
    
    return None, None, None, None, None

def calculer_distance_metres(lat1, lon1, lat2, lon2): #J'ai gardé une version python, plus rapide pour les calculs d'après
    R = 6371000.0  # Rayon de la Terre en mètres
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)

    # Formule de Haversine
    a = np.sin(d_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    
    return R * c
    
#------Formulaire utilisateur------

col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 2])

with col_a:
    adresse_input = st.text_input("Adresse exacte du bien :", "12 Rue de la Paix 75002 Paris")
with col_b:
    type_bien = st.selectbox("Type local :", ["Appartement", "Maison"])
with col_c:
    surface_input = st.number_input("Surface (m²) :", min_value=10, max_value=500, value=60)
with col_d:
    rayon_m = st.slider("Rayon (mètres) :", min_value=50, max_value=3000, value=200, step=50)

if adresse_input:
    lat, lon, label_adresse, code_commune, commune = geocode_adresse(adresse_input) #API nous renvoie coord GPS et code commune INSEE
    
    if lat and lon and code_commune:
        st.success(f"📍 **Bien localisé :** {label_adresse} (Code INSEE : `{code_commune}`)")
        
        df_zone = get_zone_sales(
            code_commune, type_bien, lat, lon, rayon_m
        )
        
        if not df_zone.empty:

            prix_m2_median = df_zone['Price per surface'].median()
            estimation_mediane = prix_m2_median * surface_input
            
            p25 = df_zone['Price per surface'].quantile(0.25) * surface_input
            p75 = df_zone['Price per surface'].quantile(0.75) * surface_input
            

            
            st.subheader(f"Estimation dans un rayon de {rayon_m} mètres")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Prix estimé (médian)", f"{estimation_mediane:,.0f} €".replace(",", " "))
            m2.metric("Prix au m² médian", f"{prix_m2_median:,.0f} €/m²".replace(",", " "))
            m3.metric("Fourchette basse (25%)", f"{p25:,.0f} €".replace(",", " "))
            m4.metric("Fourchette haute (75%)", f"{p75:,.0f} €".replace(",", " "))
            
            st.caption(f"Calcul réalisé sur **{len(df_zone)} transactions réelles** de type '{type_bien}' recensées dans un rayon de **{rayon_m}m** entre 2021 et 2025.")
            

            st.write("")
            col_graph, col_table = st.columns([1, 1])
            
            with col_graph:
                with st.expander("Évolution du prix/m² dans la zone"):
                    df_trend = df_zone.groupby('year')['Price per surface'].median().reset_index()
                    df_trend['year'] = df_trend['year'].astype(str)
                    df_trend = df_trend.rename(columns={'year': 'Année', 'Price per surface': 'Prix médian (€/m²)'})
                    st.line_chart(df_trend.set_index('Année'))
            
            with col_table:
                with st.expander(f"Ventes à moins de {rayon_m}m"):
                    df_display = df_zone.sort_values(by='distance_m', ascending=True).copy()
                    df_display['distance_m'] = df_display['distance_m'].round(0)
                    
                    cols_to_show = ['distance_m', 'year', 'Valeur fonciere', 'Surface reelle bati', 'Price per surface', 'Nombre pieces principales']
                    st.dataframe(
                        df_display[cols_to_show],
                        column_config={
                            "distance_m": st.column_config.NumberColumn("Distance", format="%d m"),
                            "year": st.column_config.NumberColumn("Année", format="%d"),
                            "Valeur fonciere": st.column_config.NumberColumn("Prix (€)", format="%d €"),
                            "Surface reelle bati": st.column_config.NumberColumn("Surface", format="%d m²"),
                            "Price per surface": st.column_config.NumberColumn("Prix/m²", format="%.0f €/m²"),
                            "Nombre pieces principales": st.column_config.NumberColumn("Pièces", format="%d"),
                        },
                        use_container_width=True,
                        height=350)
                
        else:
            st.warning(f"Aucune transaction pour un(e) **{type_bien}** n'a été trouvée dans un rayon de **{rayon_m}m** autour de cette adresse. Augmentez le rayon avec le curseur.")
    
    else:
        st.error("Adresse introuvable. Précisez le numéro, la rue et la ville.")

#-------------------------------------------
#----Informations autour de l'adresse-------
#-------------------------------------------

st.divider()
st.markdown(f"## Informations autour de {commune}")


lat, lon, label_adresse, code_commune, commune = geocode_adresse(adresse_input)

#----API transports en commun----

url = "https://transport.data.gouv.fr/api/gtfs-stops"

def point_to_bbox(lat: float, lon: float, radius_km: float):

    delta_lat = ((radius_km)/1000) / 111.0
    delta_lon = ((radius_km)/1000) / (111.0 * math.cos(math.radians(lat)))
    
    min_lat = lat - delta_lat
    max_lat = lat + delta_lat
    min_lon = lon - delta_lon
    max_lon = lon + delta_lon
    
    return min_lat, max_lat, min_lon, max_lon

min_lat, max_lat, min_lon, max_lon = point_to_bbox(lat, lon, rayon_m)

params = {
    "south": min_lat,  
    "north": max_lat,  
    "west": min_lon,   
    "east": max_lon    
}

try:

    response = requests.get(url, params = params)
    
    if response.status_code == 200:
        data = response.json()
    
        stops = data.get("features", [])
    
        table_data = []
        for item in stops:
            props = item.get("properties", {})
            coords = item.get("geometry", {}).get("coordinates", [])
            if coords:
                table_data.append({
                    "Nom de l'arrêt": props.get("stop_name"),
                    "ID": props.get("dataset_title"),
                    "latitude": coords[1],  
                    "longitude": coords[0]
                })
            
        
        df = pd.DataFrame(table_data)
    
        df = df.drop_duplicates(subset=["latitude"])
except requests.exceptions.Timeout:
    logger.warning("Timeout lors du chargement des transports")
    df = pd.DataFrame()
    st.warning("Délai d'attente dépassé pour les transports en commun.")
except Exception as e:
    logger.error(f"Erreur transports: {type(e).__name__}")
    df = pd.DataFrame()
    st.warning("Impossible de charger les données de transport.")


#---API Gare ferroviaire----

url = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/liste-des-gares/records"

params = {
    "where": f"within_distance(c_geo, geom'POINT({lon} {lat})', {rayon_m}m)",
    "limit": 50
}

params_plus_proche = {
    "order_by": f"distance(c_geo, geom'POINT({lon} {lat})')",
    "limit": 1
}
try:
    response = requests.get(url, params=params)
    response_plus_proche = requests.get(url, params=params_plus_proche)
    
    #Toutes les gares
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
    
        gares_list = []
        for g in results:
            val_voyageurs = str(g.get("voyageurs", "")).strip().upper()
            if val_voyageurs in ["O", "OUI", "TRUE", "1"]:
                pos = g.get("c_geo", {}) or {}
                gares_list.append({
                    "Gare": g.get("libelle"),
                    "latitude": pos.get("lat"),
                    "longitude": pos.get("lon"),
                    "Voyageurs": True
                })
    
        df_gares = pd.DataFrame(gares_list)
    
    #Gare la plus proche
    gare_la_plus_proche = None
    
    lat_origine = lat  
    lon_origine = lon  
    
    if response_plus_proche.status_code == 200:
        data = response_plus_proche.json()
        results = data.get("results", [])
    
        for g in results:
    
            fields = g.get("fields", g)
    
            val_voyageurs = str(fields.get("voyageurs", "")).strip().upper()
    
            if val_voyageurs in ["O", "OUI", "TRUE", "1"]:
                pos = fields.get("c_geo") or {}
    
    
                if isinstance(pos, dict):
                    lat_g, lon_g = pos.get("lat"), pos.get("lon")
                elif isinstance(pos, list) and len(pos) == 2:
                    lat_g, lon_g = pos[0], pos[1]
                else:
                    lat_g, lon_g = None, None
    
                gare_la_plus_proche = {
                    "Gare": fields.get("libelle"),
                    "latitude": lat_g,
                    "longitude": lon_g,
                    "Voyageurs": True,
                }
                break  
except requests.exceptions.Timeout:
    logger.warning("Timeout lors du chargement des gares")
    df_gares = pd.DataFrame()
    gare_la_plus_proche = None
    st.warning("Délai d'attente dépassé pour les gares ferroviaires.")
except Exception as e:
    logger.error(f"Erreur gares: {type(e).__name__}")
    df_gares = pd.DataFrame()
    gare_la_plus_proche = None


if gare_la_plus_proche and gare_la_plus_proche["latitude"] is not None:
    lat_gare = gare_la_plus_proche["latitude"]
    lon_gare = gare_la_plus_proche["longitude"]


    distance_gare_la_plus_proche = (
        calculer_distance_metres(
            lat_origine, lon_origine, lat_gare, lon_gare
        )
        / 1000
    )
else:
    lat_gare = None
    lon_gare = None
    distance_gare_la_plus_proche = None


    

#----API Risque-------

url = "https://georisques.gouv.fr/api/v1/resultats_rapport_risque"
params = {
    "latlon": f"{lon},{lat}"}


try:
    response = requests.get(url, params=params, timeout=5)
    response.raise_for_status()
    
    if response.status_code == 200:
        data = response.json()
        communes = data.get("lesRisquesParCommune", [])
    else:
        data = {}
        st.warning("Impossible de charger les données de risque.")

except requests.exceptions.Timeout:
    logger.warning("Timeout lors du chargement des risques")
    data = {}
    st.warning("Délai d'attente dépassé pour les risques.")
except Exception as e:
    logger.error(f"Erreur risques: {type(e).__name__}")
    data = {}




#-----Mise en forme--------

col1, col2, col3 = st.columns([2,2,2])

with col1:

    with st.container(border=True, height = 450, width = 480):

        st.subheader("Transport en commun")

        st.write(f"{len(df)} arrêts trouvés dans le périmètre sélectionné.")

        tab1, tab2, tab3 = st.tabs(["Transport en commun", "Gare ferroviaire", "Liste"])
    
        with tab1:
            if len(df) == 0:
                st.warning("Pas de transport en commun dans le périmètre sélectionné")
            else:
                st.map(df, latitude="latitude", longitude="longitude", size=10, width = 440, height = 260, zoom = 15)

        with tab2:
            if len(df_gares) == 0:
                st.warning("Pas de gare voyageur trouvée dans le périmètre sélectionné.")
        
                if gare_la_plus_proche and distance_gare_la_plus_proche is not None:
                    nom_gare = gare_la_plus_proche.get("Gare", "Inconnue")
                    st.write(
                        f"**Gare la plus proche :** {nom_gare} à"
                        f" {distance_gare_la_plus_proche:.1f} km"
                    )
            else:
                st.map(df_gares, latitude="latitude", longitude="longitude", size=10, width = 440, height = 260, zoom = 15)
                              
        with tab3:
            if len(df) == 0:
                st.warning("Pas de transport en commun dans le périmètre sélectionné")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

with col2:
    with st.container(border=True, height=450):
        
        if "risquesNaturels" in data or "risquesTechnologiques" in data:
            

            st.subheader("Risques naturels sur la commune")
            
            risques_nat = data.get("risquesNaturels", {})
            
            risques_nat_actifs = []

            for risque in risques_nat.values():
                if risque.get("present") == True:
                    risques_nat_actifs.append(risque)
            
            if risques_nat_actifs:
                for risque in risques_nat_actifs:
                    nom = risque.get("libelle", "Risque inconnu")
                    statut = risque.get("libelleStatutCommune") or "Statut inconnu"
                    
                    st.warning(f"• **{nom}** \n\n_{statut}_")
            else:
                st.success("Aucun risque naturel majeur recensé.")

        
            st.subheader("Risques technologiques")
            
            risques_tech = data.get("risquesTechnologiques", {})
            
            risques_tech_actifs = []

            for risque in risques_tech.values():
                if risque.get("present") == True:
                    risques_tech_actifs.append(risque)
            
            if risques_tech_actifs:
                for risque in risques_tech_actifs:
                    nom = risque.get("libelle", "Risque inconnu")
                    statut = risque.get("libelleStatutCommune") or "Statut inconnu"
                    
                    st.error(f"• **{nom}** \n\n_{statut}_")
            else:
                st.success("Aucun risque technologique majeur recensé.")
        else:
            st.info("Aucune donnée de risque disponible pour cette adresse.")
            
        
            
