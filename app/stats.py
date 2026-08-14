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
import duckdb
import statsmodels


st.title("Statistiques détaillées", text_alignment = 'center')
st.markdown("## Analyse du marché immobilier", text_alignment = 'center')

st.divider()



#----------------------------------------
#-----Test de Kolmogorov-Smirnov--------- 
#----------------------------------------

st.subheader('1) Normalité des données : test de Kolmogorov-Smirnov')
st.markdown("#### Contexte")
st.write("Dans le cadre de l'analyse des fichiers DVF, la modélisation et le choix des tests statistiques reposent sur la condition de normalité des variables explicatives.\n\n **Les prix immobiliers ainsi que les surfaces, représentent une asymétrie importante sur la droite** (distributions étalées vers les valeurs élevées et présence de valeurs aberrantes). Afin de déterminer si la distribution suit une loi normale, un test d'ajustement de Kolmogorov-Smirnov a était réalisé. Ce test non paramétrique permet de **comparer la fonction de distribution cumulative empirique des données observées à la fonction de distribution cumulative théorique d'une loi normale de même moyenne et de même variance.**") 
st.write(" \n **Note:** Le test ici est réalisé uniquement à des fins académiques ; le test sur le fichier DVF a peu d'intérêt puisque nous disposons d'une base de données de plusieurs million de lignes. La p-value sera toujours égale à 0 car le test sera très sévère sur les écarts extrèmes.")
st.markdown("#### Hypothèses")
st.write("**Hypothèse nulle $H_0$**: La variable suit une loi normale. \n\n Distribution observée = $N$($μ$,$σ$²)")
st.write("**Hypothèse alternative $H_1$** : La variable ne suit pas une loi normale. \n\n Distribution observée ≠ $N$($μ$,$σ$²)")

#--Requete SQL pour les sample---

@st.cache_data
def get_sample_data():
    connexion = duckdb.connect()

    requete_1 = f"""
    SELECT "Valeur fonciere"
    FROM 'dvf_clean_2021_2025_v2.parquet'
    USING SAMPLE 100
    """
    sample = connexion.execute(requete_1).df()

    requete_2 = f"""
    SELECT "Price per surface"
    FROM 'dvf_clean_2021_2025_v2.parquet'
    USING SAMPLE 100
    """
    sample_m2 = connexion.execute(requete_2).df()

    connexion.close()
    return sample, sample_m2

sample, sample_m2 = get_sample_data()

sample_mean = sample['Valeur fonciere'].mean()
sample_std = sample['Valeur fonciere'].std()

sample_mean_m2 = sample_m2['Price per surface'].mean()
sample_std_m2 = sample_m2['Price per surface'].std()

stat2, p_value2 = kstest(
    sample['Valeur fonciere'],
    'norm',
    args=(sample_mean, sample_std),)

stat3, p_value3 = kstest(
    sample_m2['Price per surface'],
    'norm',
    args=(sample_mean_m2, sample_std_m2),)

st.markdown("#### **Test sur la valeur foncière**")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label = 'Statistique KS (max D)',
        value = round(stat2, 2),
        delta = 'Ecart test|loi normale (0 à 1)',
        delta_color = 'inverse')
with col2:
    st.metric(
        label = '$p$_value',
        value = round(p_value2, 2),
        delta = 'Rejet de $H$0 (<0.05)',
        delta_color = 'inverse')        

col1, col2 = st.columns(2)

with col1:

    fig, ax1 = plt.subplots(figsize=(7,3), facecolor = 'white')
    
    sns.kdeplot(
        data=sample['Valeur fonciere'] / 1000,
        color='forestgreen', 
        lw=3,                
        ax=ax1
    )
    
    ax1.set_title('Distribution brute')
    ax1.set_xlabel('Prix en k€')
    ax1.set_ylabel('Densité')

    plt.suptitle('Distribution des prix de vente')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:

    fig, ax2 = plt.subplots(figsize=(7,3), facecolor = 'white')

    sns.kdeplot(
        data=np.log(sample['Valeur fonciere']),
        color='red', 
        lw=3,                
        ax=ax2
    )
    
    ax2.set_title('Log-transform distribution')
    ax2.set_xlabel('Log(price)')
    ax2.set_ylabel('Densité (log)')

    plt.suptitle('Distribution des prix de vente')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

#-----Price per m²-------

st.write('**Test sur le prix au m²**')

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label = 'Statistique KS (max D)',
        value = round(stat3, 2),
        delta = 'Ecart test|loi normale (0 à 1)',
        delta_color = 'inverse')
with col2:
    st.metric(
        label = '$p$_value',
        value = round(p_value3, 2),
        delta = 'Rejet de $H$0 (<0.05)',
        delta_color = 'inverse')

col1, col2 = st.columns(2)

with col1:

    fig, ax3 = plt.subplots(figsize=(7,3), facecolor = 'white')
    
    sns.kdeplot(
        data=sample_m2['Price per surface'],
        color='forestgreen', 
        lw=3,                
        ax=ax3
    )
    
    ax3.set_title('Distribution brute')
    ax3.set_xlabel('Prix en €')
    ax3.set_ylabel('Densité')

    plt.suptitle('Distribution du prix par m²')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
with col2:

    fig, ax4 = plt.subplots(figsize=(7,3), facecolor = 'white')
    
    sns.kdeplot(
        data=np.log(sample_m2['Price per surface']),
        color='red', 
        lw=3,                
        ax=ax4
    )
    
    ax4.set_title('Log-transform distribution')
    ax4.set_xlabel('Log(price)')
    ax4.set_ylabel('Densité (log)')

    plt.suptitle('Distribution du prix par m²')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.success("""
**💡 Conclusion statistique & impact métier :**
* **Absence de normalité brute :** Les tests de Kolmogorov-Smirnov et les courbes vertes confirment que les prix ne suivent pas une loi normale (forte asymétrie vers les valeurs élevées).
* **Effet Log-normal :** Une fois transformés par logarithme (courbes rouges), les prix se rapprochent d'une distribution normale (loi log-normale).
* **Recommandation :** Puisque les données brutes ne sont pas normales, **la moyenne et l'écart-type ne sont pas adaptés**. Il faut impérativement privilégier la **médiane** et l'**écart interquartile (IQR)** pour éviter d'être biaisé par les biens d'exception.
""")

st.divider()

#-------------------------------------------------------
#----------Test de Mann-Whitney rural vs urbain---------
#-------------------------------------------------------

st.subheader("2) Analyse comparative : département urbain vs rural")
st.markdown("#### Contexte")
st.write("Dans le cadre de cette analyse, nous cherchons à déterminer s'il existe une différence significative de niveau de prix entre les départements dits urbains et les départements dits ruraux (source: INSEE).\n\n La distributivité présentant un fort écart de distributivé à la normalité (déterminé en amont par le test de Kolomorov-Smirnov), **nous ne pourrons pas utiliser un test t student paramétrique**. Nous avons donc retenu le test non paramétrique de Mann-Whitney pour échantillons indépendants.")

st.markdown("#### Hypothèses")
st.write("**Hypothèse nulle $H_0$**: Il n'existe pas de différence de prix entre les deux types de département. \n\n")
st.write(r"$$P(X_{\text{urbain}} > Y_{\text{rural}}) = 0.5$$")
st.write("**Hypothèse alternative $H_1$** : Il existe une différence de prix entre les deux types de département. \n\n")
st.write(r"$$P(X_{\text{urbain}} > Y_{\text{rural}}) ≠ 0.5$$")

st.markdown("#### Résultats du test")

# --- 1. CALCULS STATISTIQUES ---

# On prend l'échantillon et on nettoie si besoin

@st.cache_data
def sample_groupe():
    requete_sql = f"""
    SELECT
        "LIBDENS",
        "Price per surface"
    FROM 'dvf_clean_2021_2025_v2.parquet'
    USING SAMPLE 100
    """
    connexion = duckdb.connect()
    sample_groupe = connexion.execute(requete_sql).df()
    connexion.close()

    return sample_groupe

groupe = sample_groupe()

groupe = groupe[['LIBDENS', 'Price per surface']]
groupe = groupe[groupe['LIBDENS'].isin(['Département à prédominance rurale', 'Département à prédominance urbaine'])]

# Attribuer les rangs
groupe['rang'] = groupe['Price per surface'].rank(method='average')
sum_rangs = groupe.groupby('LIBDENS')['rang'].sum()
sum_urbain = sum_rangs['Département à prédominance urbaine']
sum_rurale = sum_rangs['Département à prédominance rurale']

count = groupe.groupby('LIBDENS')['rang'].count()
n1 = count['Département à prédominance urbaine']
n2 = count['Département à prédominance rurale']

# Statistiques U
U1 = n1 * n2 + (n1 * (n1 + 1)) / 2 - sum_urbain
U2 = n1 * n2 + (n2 * (n2 + 1)) / 2 - sum_rurale
u_final = min(U1, U2)

mu_U = (n1 * n2) / 2
sigma_U = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
Z = (u_final - mu_U) / sigma_U
p_value = 2 * (1 - norm.cdf(abs(Z)))

#----Méthode automatique----

urbain = groupe[groupe['LIBDENS'] == 'Départements à prédominance urbain']['Price per surface']
rurale = groupe[groupe['LIBDENS'] == 'Départements à prédominance rurale']['Price per surface']
stat, p_value3 = mannwhitneyu(urbain, rurale, alternative = 'two-sided') #Obtention de la même valeur, check !

# Taille d'effet (r de Rosenthal)
N = len(groupe)
r_effect_size = abs(Z) / (N**0.5)


# --- 2. AFFICHAGE DES RÉSULTATS EN KPIs (MÉTRIQUES) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Score $Z$",
        value=f"{Z:.1f}",
        delta="Seuil critique > 1.96",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="$p$-value",
        value=f"{p_value:.1e}" if p_value < 0.0001 else f"{p_value:.4f}",
        delta="Rejet de $H_0$ (< 0.05)",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Taille d'effet (r)",
        value=f"{r_effect_size:.4f}",
        delta="Effet très fort (> 0.5)",
        delta_color="normal"
    )


st.success("""
**💡 Interprétation des résultats :**
* **Rejet de l'hypothèse nulle ($H_0$) :** Le score $|Z|$ est largement supérieur à 1.96 et la $p$-value est quasi nulle (inférieur au seuil alpha de 5%)
* **Conclusion :** Nous pouvons affirmer avec certitude que les prix au m² dans les départements urbains sont **très largement supérieurs** à ceux des zones rurales, avec une **influence très forte** de la catégorie du département.
""")

st.divider()


# --- 3. GRAPHIQUE COTE COTE ---
col1, col2 = st.columns([0.7, 1])

groupe['LIBDENS'] = groupe['LIBDENS'].astype(str)

with col1: 

    valeurs_reelles_libdens = ["Département à prédominance urbaine", "Département à prédominance rurale"]
    
    labels_modifies = [
        "Département à\nprédominance urbain",
        "Département à\nprédominance rurale"
    ]

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='white')
    sns.set_style("whitegrid")
    
    sns.boxplot(
        data=groupe,
        x='LIBDENS',
        y='Price per surface',
        width=0.5,
        palette=['#D9D2FC', '#8ecae6'],
        ax=ax,
        order=valeurs_reelles_libdens,
        showfliers=False,
        hue='LIBDENS',
        legend=False,
    )

    ax.set_xticklabels(labels_modifies, fontsize=9)
    ax.set_xlabel('Catégorie de département', fontweight='bold', fontsize=10)
    ax.set_ylabel('Prix au m² (€)', fontweight='bold', fontsize=10)
    ax.set_title('Distribution des prix par zone', fontweight='bold', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.markdown("### Méthodologie statistique")
    
    with st.expander("Voir les détails techniques du test", expanded = True):
        st.write(f"""
        - **Test appliqué :** U de Mann-Whitney bilatéral (manuel & validé par `scipy`).
        - **Effectif total ($N$) :** {N:,} transactions analysées.
        - **Somme des rangs Urbain :** {sum_urbain:,.0f} ($n_1 = {n1:,}$)
        - **Somme des rangs Rurale :** {sum_rurale:,.0f} ($n_2 = {n2:,}$)
        - **Statistique U finale :** {u_final:,.1f}
        """)
        
    st.info("Le graphique (sans les valeurs extrêmes/outliers pour plus de lisibilité) met en évidence l'écart significatif entre les médianes des deux populations.")

#---------------------------------------------------------------------------------------
#------Test de différence de prix entre les types de rues (test de Kruskal-Wallis)------
#---------------------------------------------------------------------------------------

st.write("")
st.divider()

st.subheader('3) Analyse comparative : différence de prix par type de voie')
st.markdown("#### Contexte")
st.write("Dans le cadre de cette analyse micro-géographique, nous cherchons à déterminer s'il existe une différence significative de niveau de prix entre les types de voie.\n\n La variable dépendante (le prix) comportant plus de deux groupes indépendants et ne respectant pas les conditions de normalité requises pour une ANOVA classique, nous avons opté pour le test non paramétrique de Kruskal-Wallis.")

st.markdown("#### Hypothèses")
st.write("**Hypothèse nulle $H_0$**: La distribution des prix est identique pour tous les types de voies. \n\n")
st.write(r"$$\mu_{\text{rang, groupe } 1} = \mu_{\text{rang, groupe } 2} = \cdots = \mu_{\text{rang, groupe } k}$$")
st.write("**Hypothèse alternative $H_1$** : La distribution des prix n'est pas identique pour tous les types de voies. \n\n")
st.write(r"$$\mu_{\text{rang, groupe } 1} ≠ \mu_{\text{rang, groupe } 2} ≠ \cdots ≠ \mu_{\text{rang, groupe } k}$$")

st.markdown("#### Résultats du test")

@st.cache_data
def groupe_kw_sample():
    requete_sql = f"""
    
    SELECT 
        "Type de voie",
        "Price per surface"
    FROM 'dvf_clean_2021_2025_v2.parquet'
    WHERE "Type de voie" IN ('RUE', 'AV', 'BD')
    USING SAMPLE 5000
    """

    connexion = duckdb.connect()
    groupe_kw_sample = connexion.execute(requete_sql).df()
    connexion.close()

    return groupe_kw_sample

groupe_KW = groupe_kw_sample()

groupe_KW['rang'] = groupe_KW['Price per surface'].rank(method='average')

sum_rang_KW = groupe_KW.groupby('Type de voie')['rang'].sum()
mean_rang_KW = groupe_KW.groupby('Type de voie')['rang'].mean()
count_rang_KW = groupe_KW.groupby('Type de voie')['rang'].count()

mean_rang_list = mean_rang_KW.tolist() #Mettre sous forme de liste pour automatiser la formule ensuite avec la somme
count_rang_list = count_rang_KW.tolist()

degree_of_freedom = 3 - 1 #Nombre de groupe - 1
n_KW = groupe_KW['rang'].count()
rank_variance = ((n_KW**2) - 1 ) / 12 #(n²-1)/12
expected_value_rank = (n_KW + 1) / 2

total_1 = (12 / (n_KW * (n_KW + 1))) #Début de la formule
total_2 = 0 #Suite de la formule

for mean, count in zip(mean_rang_list, count_rang_list):
    total_2 += (count * ((mean - expected_value_rank)**2))

test_H = total_1 * total_2
seuil_critique = 5.99 #Lecture de la table de chi-deux

p_value6 = 1 - scipy_stats.chi2.cdf(test_H, df=degree_of_freedom)

#----Effect size pour le test (epsilon²)--------

epsilon = test_H / (n_KW - 1)

#-----Méthode automatique------

stats, p_value7 = kruskal(
    groupe_KW[groupe_KW['Type de voie'] == 'RUE']['Price per surface'],
    groupe_KW[groupe_KW['Type de voie'] == 'AV']['Price per surface'],
    groupe_KW[groupe_KW['Type de voie'] == 'BD']['Price per surface'],
)

# 1. AFFICHAGE DES RÉSULTATS EN MÉTRIQUES

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label = 'Test H',
        value = f'{test_H:.1f}',
        delta = "α = 0.05: Seuil critique > 5.99",
        delta_color = "normal"
    )
with col2:
    st.metric(
        label = "$p$_value",
        value=f"{p_value6:.1e}" if p_value < 0.0001 else f"{p_value6:.4f}",
        delta="Rejet de H0 (< 0.05)",
        delta_color="inverse"
    )
with col3:
    st.metric(
        label = "Taille d'effet ($\epsilon^2$)",
        value = f'{round(epsilon,2)}',
        delta = "Effet faible",
        delta_color="normal"
    )

st.success("""
**💡 Interprétation des résultats :**
* **Rejet de l'hypothèse nulle ($H_0$) :** Le score $H$ est largement supérieur au seuil critique de 5.99 et la $p$-value est quasi nulle.
* **Conclusion:** Nous pouvons affirmer avec certitude qu'il existe belle et bien une différence de prix au m² entre les différents types de voie.
Cependant, grâce à l'analyse de la taille d'effet, nous pouvons dire que le type de voie n'est responsable que d'environ 2% de la variabilité des prix.""")

st.divider()

# 2.MISE EN PAGE EN DESSOUS

col1, col2 = st.columns([0.7, 1])

groupe_KW['Type de voie'] = groupe_KW['Type de voie'].astype(str)

with col1:
    fig, ax = plt.subplots(figsize = (4,4), facecolor = 'white')
    
    sns.set_style('whitegrid')
    sns.boxplot(
        x='Type de voie',
        y= 'Price per surface',
        data = groupe_KW,
        width = 0.4,
        hue = 'Type de voie',
        palette=['#D9D2FC', '#8ecae6', '#EB958A'],
        ax=ax,
        showfliers = False,
        legend = False)
    
    ax.set_ylabel('Prix médian au m² (en €)', fontweight='bold')
    ax.set_xlabel('Type de voie', fontweight='bold')
    ax.set_title('Distribution des prix par type de voie', fontweight = 'bold', fontsize = 14)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.markdown("### Méthodologie statistique")

    with st.expander('Voir les détails techniques du test', expanded = True):
        st.write(f"""
        - **Test appliqué :** H de Kruskal-Wallis (manuel & validé par `scipy`).
        - **Effectif total ($N$) :** {n_KW:,} transactions analysées.
        - **Somme des rangs *rue* :** {sum_rang_KW.loc['RUE']:,.0f} ($n_1 = {count_rang_KW.loc['RUE']:,}$)
        - **Somme des rangs *avenue* :** {sum_rang_KW.loc['AV']:,.0f} ($n_2 = {count_rang_KW.loc['AV']:,}$)
        - **Somme des rangs *boulevard* :** {sum_rang_KW.loc['BD']:,.0f} ($n_3 = {count_rang_KW.loc['BD']:,}$)
        - **Degré de liberté :** {degree_of_freedom}
        - **Variance de rang :** {rank_variance}
        """)
    
st.info("Le graphique (sans les valeurs extrêmes/outliers pour plus de lisibilité) met en évidence l'écart significatif entre les médianes des trois distributions.")
    
st.divider()



#--------------------------------------------------------------------------------------------
#------Test de Dunn-Bonferroni (différencier les paires qui bougent par type de voie)--------
#--------------------------------------------------------------------------------------------

#Comme nous l'avons vu avec le test de Kruskal-Wallis, il existe bien une différence de prix entre les différents types de voie.
#L'objectif de ce test est de déterminer les différences entre chaque paire. Et oui ! Pour l'instant nous ne savons pas si le BD est plus cher qu'une rue par exemple :)

groupes_name = mean_rang_KW.index.tolist() #Récupérer les noms des types de voie
N = n_KW

#Nombre de comparaison possible
k =len(groupe)
num_comparaison = k * (k-1) / 2


dunn_results = {}

#Boucle sur toutes les paires possible de groupes
for g1, g2 in itertools.combinations(groupes_name, 2):
    mean1 = mean_rang_KW[g1]
    mean2 = mean_rang_KW[g2]
    n1 = count_rang_KW[g1]
    n2 = count_rang_KW[g2]

    std_error = np.sqrt(((N * (N + 1)) / 12) * ((1 / n1) + (1 / n2)))

    z_stat = (mean1 - mean2) / std_error

    p_val_brute = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

    # Correction de Bonferroni (on multiplie la p-value par le nombre de comparaisons)
    p_val_adjusted = min(p_val_brute * num_comparaison, 1.0)

    dunn_results[f'{g1} vs {g2}'] = {
        'z_stat': z_stat,
        'p_value_brute': p_val_brute,
        'p_value_adjusted': p_val_adjusted}

for pair, res in dunn_results.items():
    if res['p_value_adjusted'] < 0.05:
        res['commentaire'] = 'Différence de prix significative (p < 0.05)'
    else:
        res['commentaire'] = 'Aucune différence significative (p > 0.05)'

data_tableau = []

for pair, res in dunn_results.items():
    data_tableau.append(
        {
        "Paire de groupes": pair,
        "Statistique z": res["z_stat"],
        "P-value ajustée": res["p_value_adjusted"],
        "Commentaire": res["commentaire"],
        })

df_dunn = pd.DataFrame(data_tableau)

st.subheader('Résultats du test post-hoc de Dunn (avec correction de Bonferroni)')

col1, col2 = st.columns([2,1])

with col1:
    st.dataframe(
            df_dunn,
            column_config = {
                "Statistique z": st.column_config.NumberColumn(
                "Statistique Z", format="%.4f"),
            "P-value ajustée": st.column_config.NumberColumn(
                "P-value (ajustée)", format="%.2e"),},
        hide_index=True,
        use_container_width=True,)
with col2:   
    df_rangs = (mean_rang_KW.sort_values(ascending=False).reset_index())
    df_rangs.columns = ["Type de voie", "Moyenne des rangs"]
    st.dataframe(df_rangs, hide_index=True, use_container_width=True)

st.warning("""
**💡 Note méthodologique :**  
Comme évoqué, la correction de Bonferroni peut parfois saturer les bornes des p-values sur de grands échantillons (ici sur BD vs RUE), mais l'analyse croisée des scores $Z$ et des rangs permet de valider avec robustesse les différences réelles entre les types de voies.
""")

#--------------------------------------------------------------------
#-------Est-ce qu'un bien avec plus de pièces vaut plus cher ?-------
#--------------------------------------------------------------------

#Utilisation du test de Kruskal-Wallis pour détermination.

st.divider()

st.subheader("4) Un bien avec plus de pièces est-il plus cher ?")
st.markdown("#### Contexte")
st.write("Dans le cadre de cette analyse, nous cherchons à déterminer si suivant le nombre de pièces principales dans un bien, il existe une influence significative sur le prix.\n\n La variable dépendante (le prix) comportant bien plus de deux groupes indépendants et ne respectant pas les conditions de normalité requises pour une ANOVA classique, nous avons opté pour le test non paramétrique de Kruskal-Wallis.")

st.markdown("#### Hypothèses")
st.write("**Hypothèse nulle $H_0$**: La distribution des prix est identique, peu importe le nombre de pièces principales. \n\n")
st.write(r"$$\mu_{\text{rang, groupe } 1} = \mu_{\text{rang, groupe } 2} = \cdots = \mu_{\text{rang, groupe } k}$$")
st.write("**Hypothèse alternative $H_1$** : La distribution des prix n'est pas identique, peu importe le nombre de pièces principales. \n\n")
st.write(r"$$\mu_{\text{rang, groupe } 1} ≠ \mu_{\text{rang, groupe } 2} ≠ \cdots ≠ \mu_{\text{rang, groupe } k}$$")

st.markdown("#### Résultats du test")

@st.cache_data
def groupe_kw_2_sample():
    requete_sql = f"""
    SELECT 
        "Nombre pieces principales",
        "Price per surface"
    FROM 'dvf_clean_2021_2025_v2.parquet'
    USING SAMPLE 5000
    """
    connexion = duckdb.connect()
    groupe_kw_2_sample = connexion.execute(requete_sql).df()
    connexion.close()

    return groupe_kw_2_sample

groupe_KW_2 = groupe_kw_2_sample()

n_KW_2 = groupe_KW_2['Price per surface'].count()

groupes = []
nb_groupe = groupe_KW_2.groupby('Nombre pieces principales', observed=True).count().reset_index()
nb_groupe = nb_groupe['Nombre pieces principales'].count() #Nombre de groupe pour calculer le seuil critique avec chi-deux table

for _, group in groupe_KW_2.groupby('Nombre pieces principales', observed=True)['Price per surface']:
    groupe_clean = group.dropna()
    groupes.append(groupe_clean)

resultat_kruskal = kruskal(*groupes)

#Taille d'effet
epsilon2 = (resultat_kruskal.statistic) / (n_KW_2 - 1)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label = 'Score $H$',
        value = round(resultat_kruskal.statistic,2),
        delta = "α = 0.05: seuil critique > 23,69",
        delta_color = 'normal')
with col2:
    st.metric(
        label = "$p$_value",
        value = round(resultat_kruskal.pvalue,2),
        delta="Rejet de $H$0 (< 0.05)",
        delta_color = 'inverse')
with col3:
    st.metric(
        label = "Taille d'effet ($\epsilon^2$)",
        value = f'{round(epsilon2,2)}',
        delta = "Effet moyen à modéré",
        delta_color="normal"
    )

st.success("""
**💡 Interprétation des résultats :**
* **Rejet de l'hypothèse nulle ($H_0$) :** Le score $H$ est largement supérieur au seuil critique de 23,69 et la $p$-value est nulle.
* **Conclusion:** Nous pouvons affirmer avec certitude qu'il existe belle et bien une différence de prix en fonction du nombre de pièces principales.
Cependant, grâce à l'analyse de la taille d'effet, nous pouvons dire que le type de voie est responsable d'environ 7% de la variabilité des prix, ce qui d'après le tableau de
Jacob Cohen, représente un effet moyen à modéré.""")

#-----------------------------------------
#-----Test de corrélation Spearman--------
#-----------------------------------------

#On test si les variables prix au m² et surface du bien varient dans le même sens de manière linéaire. La base DVF contient beaucoup de valeurs extrêmes. Le test habituel de Pearson (r) exige une linéarité parfaite et des données gaussiennes, tandis que Spearman travaille sur les rangs des données et s'avère beaucoup plus résistant aux valeurs aberrantes.
st.divider()

st.subheader("5) Existe-t'il une corrélation entre le prix au m² et la surface d'un bien ?")
st.markdown("#### Contexte")
st.write("Dans le cadre de cette analyse, nous cherchons à déterminer s'il existe une corrélation entre les deux variables prix et surface. Pour ce faire, j'ai décidé de réaliser un test $r$ de Spearman, non-paramétrique. En effet, comme démontré lors du test de Kolmogorov-Smirnov, les données ne suivent pas une distribution normale, ce qui nous empêche d'utiliser le test paramétrique de Pearson.")

st.markdown("#### Hypothèses")
st.write("**Hypothèse nulle $H_0$**: Il n'y a pas de relation monotone entre le prix au m² et la surface du bien. \n\n")
st.write(r"$$\rho_s = 0$$")
st.write("**Hypothèse alternative $H_1$** : Il existe une relation monotone entre le prix au m² et la surface du bien. \n\n")
st.write(r"$$\rho_s \neq 0$$")

st.markdown("#### Résultats du test")

@st.cache_data
def test_spearman_function():
    requete_sql = f"""
    SELECT
        "Price per surface",
        "Surface reelle bati"
    FROM 'dvf_clean_2021_2025_v2.parquet'
    USING SAMPLE 1000
    """

    connexion = duckdb.connect()
    spearman_df = connexion.execute(requete_sql).df()
    connexion.close()

    return spearman_df

spearman_df = test_spearman_function()

#---Manuel-----

spearman_df = spearman_df.sort_values(by='Price per surface', ascending=False)
spearman_df['rang_prix'] = range(1, len(spearman_df) + 1)

spearman_df = spearman_df.sort_values(by='Surface reelle bati', ascending=False)
spearman_df['rang_surface'] = range(1, len(spearman_df) + 1)

taille_df = len(spearman_df)

spearman_df['d'] = spearman_df['rang_prix'] - spearman_df['rang_surface']
spearman_df['d2'] = spearman_df['d']**2

r = 1 - ((6*spearman_df['d2'].sum())/(taille_df*(((taille_df)**2)-1)))

#Le score est le même :)

#---Test auto via Scipy----

test_spearman = spearmanr(spearman_df['Price per surface'], spearman_df['Surface reelle bati'])

correlation = test_spearman.statistic
p_value = test_spearman.pvalue

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label = r"Score $$\rho_s$$",
        value = round(correlation,2),
        delta = "Score compris entre 1 et -1",
        delta_color = 'normal')
with col2:
    st.metric(
        label = "$p$_value",
        value = round(p_value,2),
        delta="Rejet de $H$0 (< 0.05)",
        delta_color = 'inverse')


st.success(r"""
**💡 Interprétation des résultats :**
* **Rejet de l'hypothèse nulle ($H_0$) :** La $p$-value est nulle.
* **Conclusion:** Suite au test de Spearman, nous pouvons conclure qu'il existe bien une corrélation modéré négative entre le prix au m² et la surface du bien. En d'autres termes, plus le bien est grand, moins le prix au m² est chere.""")

fig = px.scatter(spearman_df, x='Surface reelle bati', 
                 y='Price per surface', 
                 trendline = "lowess", 
                 trendline_color_override="red", 
                 log_y = True,
                opacity = 0.4,
                title="Évolution du prix au m² en fonction de la surface",
                labels={
        'Surface reelle bati': 'Surface (m²)',
        'Price per surface': 'Prix au m² (€)'
    })

st.plotly_chart(fig)

#---Calcul de la variation--- (j'utilise pas spearman_df car le sample est trop petit)

@st.cache_data
def test_spearman_function_bigsample():
    requete_sql = f"""
    SELECT
        "Price per surface",
        "Surface reelle bati"
    FROM 'dvf_clean_2021_2025_v2.parquet'
    USING SAMPLE 100000
    """

    connexion = duckdb.connect()
    spearman_df_bigsample = connexion.execute(requete_sql).df()
    connexion.close()

    return spearman_df_bigsample

spearman_df_bigsample = test_spearman_function_bigsample()

st.markdown("#### Evaluer la variation de prix au m² entre deux biens de surface différente")

col1, col2, col3 = st.columns([1,1,2])

with col1:
    surface_1 = st.number_input("Surface du bien", value = 30)
with col2:
    surface_2 = st.number_input("Surface du bien à comparer", value = 40)

surface_1_min = surface_1 * 0.95
surface_1_max = surface_1 * 1.05

filtre_surface_1 = (spearman_df_bigsample['Surface reelle bati'] >= surface_1_min) & (spearman_df_bigsample['Surface reelle bati'] <= surface_1_max)

prix_moyen_1 = spearman_df_bigsample[filtre_surface_1]['Price per surface'].mean()

surface_2_min = surface_2 * 0.95
surface_2_max = surface_2 * 1.05

filtre_surface_2 = (spearman_df_bigsample['Surface reelle bati'] >= surface_2_min) & (spearman_df_bigsample['Surface reelle bati'] <= surface_2_max)

prix_moyen_2 = spearman_df_bigsample[filtre_surface_2]['Price per surface'].mean()

var = ((prix_moyen_2 - prix_moyen_1) / prix_moyen_1) *100

with col3:
    st.metric(label="Variation", value=f"{var:.2f}%")


#------------------------------------------
#-----NAVIGATION SUR LA PAGE (SideBar)-----
#------------------------------------------

with st.sidebar:
    st.caption("NAVIGATION SUR LA PAGE")
    st.markdown("[1. Normalité des données](#1-normalite-des-donnees-test-de-kolmogorov-smirnov)")
    st.markdown("[2. Département urbain vs rural](#2-analyse-comparative-departement-urbain-vs-rural)")
    st.markdown("[3. Écart de prix par type de voie](#3-analyse-comparative-difference-de-prix-par-type-de-voie)")
    st.markdown("[4. Écart de prix par nombre de pièces](#4-un-bien-avec-plus-de-pieces-est-il-plus-cher)")
    st.markdown("[5. Corrélation entre le prix au m² et la surface](#5-existe-t-il-une-correlation-entre-le-prix-au-m-et-la-surface-d-un-bien)")
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