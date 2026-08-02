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




df_clean = st.session_state.df_clean

#-----------------------------------
#-------Menu des tests stats--------
#-----------------------------------

st.title("Statistiques détaillées")

#----------------------------------------
#-----Test de Kolmogorov-Smirnov--------- 
#----------------------------------------

st.subheader('1) Normalité des données : test de Kolmogorov-Smirnov')

st.write("Note: le test ici est réalisé uniquement pour raison académique, le test sur le fichier DVF n'a aucun intérêt dans le sens où nous avons un dataframe de plusieurs millions de lignes. La p-value sera toujours égale à 0 car le test sera très sévère sur les écarts extremes.")

sample = df_clean.dropna(subset=['Valeur fonciere']).sample(100, random_state=42)
sample_mean = sample['Valeur fonciere'].mean()
sample_std = sample['Valeur fonciere'].std()

sample_m2 = df_clean.dropna(subset=['Price per surface']).sample(100, random_state=42)
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

st.write('**Test sur la valeur fonciere**')

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label = 'Statistique KS (max D)',
        value = round(stat2, 2),
        delta = 'Ecart test|loi normal (0 à 1)',
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
    
    ax1.set_title('Gross distribution')
    ax1.set_xlabel('Price in k€')
    ax1.set_ylabel('Density')

    plt.suptitle('Distribution per sales price')
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
    ax2.set_ylabel('Density (log)')

    plt.suptitle('Distribution per sales price')
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
        delta = 'Ecart test|loi normal (0 à 1)',
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
    
    ax3.set_title('Gross distribution')
    ax3.set_xlabel('Price in €')
    ax3.set_ylabel('Density')

    plt.suptitle('Distribution price per m²')
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
    ax4.set_ylabel('Density (log)')

    plt.suptitle('Distribution price per m²')
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

st.subheader("1) Analyse comparative : urbain vs rural")
st.write("Test statistique de Mann-Whitney sur un échantillon aléatoire de transactions.")

# --- 1. CALCULS STATISTIQUES ---
df_clean.columns = df_clean.columns.str.strip()

# On prend l'échantillon et on nettoie si besoin
groupe = df_clean.sample(50000)
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

urbain = groupe[groupe['LIBDENS'] == 'Département à prédominance urbaine']['Price per surface']
rurale = groupe[groupe['LIBDENS'] == 'Département à prédominance rurale']['Price per surface']
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
* **Rejet de l'hypothèse nulle ($H_0$) :** Le score $|Z|$ est largement supérieur à 1.96 et la $p$-value est quasi nulle.
* **Conclusion :** Nous pouvons affirmer avec certitude que les prix au m² dans les départements urbains sont **très largement supérieurs** à ceux des zones rurales, avec une **influence très forte** de la catégorie du département.
""")

st.divider()


# --- 3. GRAPHIQUE COTE COTE ---
col1, col2 = st.columns([0.7, 1])

groupe['LIBDENS'] = groupe['LIBDENS'].astype(str)

with col1:       
    labels_modifies = [
        "Département à\nprédominance urbain",
        "Département à\nprédominance rurale",
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
        showfliers=False,
        hue='LIBDENS',
        legend=False,
    )

    ax.set_xticks(range(len(labels_modifies)))
    ax.set_xticklabels(labels_modifies, fontsize=9)
    ax.set_xlabel('Catégorie de département', fontweight='bold', fontsize=10)
    ax.set_ylabel('Prix au m² (€)', fontweight='bold', fontsize=10)
    ax.set_title('Distribution des prix par zone', fontweight='bold', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.markdown("### Méthodologie statistique")
    
    with st.expander("Voir les détails techniques du test"):
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

st.subheader('2) Analyse comparative : différence de prix sur les types de voie')
st.write('Test statistique de Kruskal-Wallis pour mettre en évidence une différence de prix sur 3 types de voie : AV, BD et RUE.')
st.write('Réalisé sur un échantillon aléatoire de transactions')

groupe_KW_transition = df_clean
groupe_KW = groupe_KW_transition[groupe_KW_transition['Type de voie'].isin(['RUE', 'AV', 'BD'])]
groupe_KW = groupe_KW.sample(5000, random_state = 42)
groupe_KW = groupe_KW[['Type de voie', 'Price per surface']]
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
        delta = "$\alpha$ = 0.05: Seuil critique > 5.99",
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

    with st.expander('Voir les détails techniques du test'):
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
Comme évoqué, la correction de Bonferroni peut parfois saturer les bornes des p-values sur de grands échantillons, mais l'analyse croisée des scores $Z$ et des rangs permet de valider avec robustesse les différences réelles entre les types de voies.
""")

#--------------------------------------------------------------------
#-------Est-ce qu'un bien avec plus de pièces vaut plus cher ?-------
#--------------------------------------------------------------------

#Utilisation du test de Kruskal-Wallis pour détermination.
#Hypothèse H0 : Il n'existe pas de différence significative des prix au mètres carré selon le nombre de pièce principales

st.divider()

st.subheader("3) Un bien avec plus de pièces est-il plus cher ?")
st.write('Test statistique de Kruskal-Wallis pour mettre en évidence une différence de prix entre les biens avec un nombre de pièces principales différent')


groupe_KW_2 = df_clean[['Nombre pieces principales', 'Price per surface']]
groupe_KW_2 = groupe_KW_2.sample(5000, random_state = 42)
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
* **Conclusion:** Nous pouvons affirmer avec certitude qu'il existe belle et bien une différence de prix en fonction du nombre de pièces principale.
Cependant, grâce à l'analyse de la taille d'effet, nous pouvons dire que le type de voie est responsable d'environ 7% de la variabilité des prix, ce qui d'après le tableau de
Jacob Cohen, représente un effet moyen à modéré.""")
