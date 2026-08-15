# Mon Analyse Immo
## Description du projet
Cette analyse est réalisée sur l'ensemble des cinq derniers fichiers DVF disponibles sur data.gouv.fr en date du 26/07/2026 ainsi que d'autres fichiers open-data afin de croiser les données.
L'objectif de cette application est double : répondre à des interrogations personnelles sur le marché de l'immobilier en France 
et mettre en pratique mes connaissances en Python et en statistiques pour tous à travers un site interactif.
## Comprendre le site
**J'ai construit l'application sur trois types d'analyse:**
- Une page d'analyse globale qui met en avant des métriques sur le marché de l'immobilier ainsi que des graphiques interactifs pour visualiser ces derniers.
- Une page de simulateur qui permet de simuler un projet immobilier en analysant des métriques et des informations sur l'environnement.
- Une page de statistiques, plus technique, qui, grâce à des tests, permet de répondre à diverses hypothèses

J'ai pris plaisir à réaliser cette analyse qui a permis de concrétiser un apprentissage continu en mathématiques et en développement.

En arrière-plan, les statistiques sont calculées de deux manières différentes : **une via la méthode manuelle afin d'appliquer les modèles mathématiques au plus proche et une en utilisant la bibliothèque *scipy* afin de vérifier et de valider mes résultats *fait main***.

## Stack technique utilisée:

- Python (3.14): Pandas, NumPy
- Visualisation: Seaborn, Matplotlib, Plotly
- Interface: Streamlit
- Statistiques: SciPy **mais surtout tout à la main !**

## Licence & Attribution

* **Code source :** Distribué sous licence AGPLv3. Voir le fichier [LICENSE] pour plus de détails.
* **Données :** Contient des données issues sous Licence Ouverte v2.0 de : 
* - DVF (DGFiP / Etalab)
  - INSEE (fichier_diffusion_2026 & BPE)
  - API data.sncf
  - API georisques.gouv.f
    
