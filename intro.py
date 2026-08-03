import streamlit as st

df_clean = st.session_state.df_clean
nb_ventes = len(df_clean)
montant_total = df_clean['Valeur fonciere'].sum() / 1e9

st.title('Introduction')

#-----------------------------------------
st.subheader("0. Informations liminaires")

st.write("Le présent jeu de données « Demandes de valeurs foncières », publié et produit par la direction générale des finances publiques, permet de connaître les transactions immobilières intervenues au cours des cinq dernières années sur le territoire métropolitain et les DOM-TOM, à l’exception de l’Alsace, de la Moselle et de Mayotte. Les données contenues sont issues des actes notariés et des informations cadastrales sur les 5 dernières années (entre 2021 et 2025)")
st.write("En application du décret du 28 décembre 2018, les informations communiquées font l’objet d’une mise à jour semestrielle. Chaque année, une première diffusion sera effectuée en avril, présentant les mutations intervenues au cours des cinq dernières années et ayant fait l’objet d’une publication par un service de publicité foncière avant le 31 décembre de l’année précédente. La diffusion du mois d’avril concerne en conséquence cinq millésimes soit 10 semestres. Une seconde diffusion sera effectuée en octobre portant sur les mutations intervenues au cours des cinq dernières années et ayant fait l’objet d’une publication par un service de publicité foncière avant le 30 juin de l’année en cours. **L’attention est appelée sur le fait qu’en avril comme en octobre, compte tenu des publications effectuées au cours du dernier semestre pouvant porter sur des mutations intervenues lors de semestres précédents, l’ensemble des fichiers annuels sont actualisés.**")
st.write("*source*: data.gouv.fr au 31/07/2026")
st.write("Le fichier release disponible via GitHub a été au maximum anonymisé pour respecter le RGPD.")

#-----------------------------------------
st.divider()
st.subheader("1. Explication de l'analyse")

st.write(
    "Cette analyse à but uniquement éducatif est réalisée sur l'ensemble "
    "des 5 derniers fichiers DVF disponibles sur data.gouv.fr en date du 26/07/2026. "
    "La construction de cette application est double : répondre à des "
    "interrogations personnelles sur le marché de l'immobilier en France et "
    "mettre en pratique mes connaissances en python et en statistiques.\n\n"

    f"📊 **Quelques chiffres clés du jeu de données nettoyé :**\n"
    f"- **{nb_ventes:,}** transactions immobilières analysées.\n"
    f"- Près de **{montant_total:.1f} milliards d'euros** de volume foncier représenté.\n"
    f"- Une couverture temporelle de **2021 à 2025**.\n\n"
    
    "**J'ai construit l'application sur deux types d'analyse:**\n"
    "- Une page d'analyse globale qui met en avant des métriques sur le marché "
    "de l'immobilier ainsi que des graphiques interactifs pour visualiser ces derniers.\n"
    "- Une page de statistiques, plus technique, qui grâce à des tests permet "
    "de répondre à diverses hypothèses.\n\n"
    "J'ai pris plaisir à réaliser cette analyse qui a permis d'enfin concrétiser "
    "un apprentissage continu en mathématiques et en développement.\n\n"
    "En fond, les statistiques sont calculées de deux manières différentes : une via la méthode manuelle afin d'appliquer les modèles mathématiques au plus proche et une en utilisant la bibliothèque *scipy* afin de vérifier et de valider mes résultats *fait main*\n\n"
    "Stack technique utilisée:\n\n"
    "- Python: Pandas, NumPy (pour la manipulation des données)\n"
    "- Visualisation: Seaborn, Matplotlib\n"
    "- Interface: Streamlit\n"
    "- Statistiques: SciPy **mais surtout tout à la main !**"
)

#-----------------------------------------
st.divider()
st.subheader("2. Database et filtres utilisés")

st.markdown("""
Afin de garantir une analyse rigoureuse et représentative du marché immobilier résidentiel français, plusieurs filtres ont été appliqués aux données brutes DVF. **La majorité proviennent directement des filtres appliqués sur le fichier officiel 'DVF : Statistiques'.**


##### 1. Périmètre de l'étude
* **Type de transaction :** Seules les **ventes réelles** (*Vente*, *Vente en l'état futur d'achèvement (VEFA)* et *Adjudication*) sont conservées. Les mutations à titre gratuit, échanges ou expropriations sont exclus car leurs montants ne reflètent pas les prix du marché libre.
* **Type de bien :** L'analyse se concentre uniquement sur les **Maisons** et **Appartements**. Les locaux commerciaux, dépendances isolées, garages et terrains nus ont été écartés.

---

##### 2. Traitement des anomalies et valeurs aberrantes (*Outliers*)
* **Seuil de prix global (> 10 000 €) :** Élimine les ventes symboliques (ex. transactions à 1 €, rachat de parts minoritaires) et les erreurs de saisie.
* **Surface habitable (entre 9 m² et 400 m²) :** 
  * Le seuil minimal de **9 m²** correspond à la surface légale minimale pour qu'un logement soit qualifié de décent en France.
  * Le plafond de **400 m²** évite d'introduire des propriétés d'exception ou d'éventuelles erreurs de saisie qui fausseraient les moyennes.
* **Prix au m² (< 100 000 €/m²) :** Suppression des ratios extrêmes dus à des erreurs administratives dans la saisie des surfaces ou des prix (recommandation *data.gouv.fr*).

---

##### 3. Isolation des ventes simples (Gestion des lots multiples)
Dans les données DVF brutes, lorsqu'un acte comporte plusieurs biens (ex. un appartement vendu avec sa cave et son parking), **la valeur foncière globale est dupliquée sur chaque ligne**.

* **Choix méthodologique :** Ne sont conservées que les mutations concernant **un unique bien**.
* **Pourquoi ?** Il est impossible de découper précisément la valeur foncière attribuée à l'appartement par rapport à ses annexes. Ce filtre évite de surestimer artificiellement les prix au m² sur les ventes par lots.
""")