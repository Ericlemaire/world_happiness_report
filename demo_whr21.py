#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:24:50 2022

@author: lemaireeric
"""

# IMPORTATION DES PACKAGES

import pandas as pd 
import numpy as np
import seaborn as sns
import streamlit as st 
import matplotlib.pyplot as plt 
import plotly.express as px 
import geopandas as gpd
import contextily as ctx
import folium,mapclassify

import plotly.figure_factory as ff
from joblib import dump, load
from xgboost import XGBRegressor

import shap 
from streamlit_shap import st_shap
shap.initjs()

import pydeck as pdk
from osgeo import gdal
#gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')



# PAGE D'ACCUEIL 


#mettre un titre
st.title("Les déterminants de la satisfaction des populations dans le monde ")
st.markdown("#### _Essai de construction d’un modèle allant au-delà d’une approche fondée sur le niveau de richesse par habitant_")
st.caption("Fait par Éric Lemaire et Rabbia Touré")
st.caption("[Profil LinkedIn d'Éric LEMAIRE](https://www.linkedin.com/in/eric-lemaire-data-scientist/)")

st.caption("[Profil LinkedIn de Rabbia TOURÉ](https://www.linkedin.com/in/rabiatouc-toure/)")

st.caption("Vous avez des suggestions d'améliorations, des idées, vous souhaitez échanger sur ce projet ? Écrivez-nous : elemaire@teamtrust.fr")
#st.image("AdobeStock_103209330.jpeg", width = 500)

st.caption("[Accéder au rapport complet du projet](https://docs.google.com/document/d/13fVdWos7I6F9gAN1kPf31MigMF7WPZyjtc3lGZM9GJY/edit?usp=sharing) ")


# CHARGEMENT ET OUVERTURE DU DATAFRAME PRINCIPAL
#whr_streamlit.csv
df = pd.read_csv('whr_streamlit.csv')
df = df.drop(['Unnamed: 0'], axis = 1)

#px = px.strip(df, x= df.columns, y= df.iloc[1, 0:])
#st.plotly_chart(px)

pages = ["Présentation du projet","Les jeux de données" ,"Explorations des données", "Présentation de la modélisation", "Interprétation des prédictions du modèle", "Conclusions et ouverture"]
page = st.sidebar.radio("Aller vers", pages)

#pages introduction
if page == pages[0]:    
    st.markdown("## Introduction")
    
    st.markdown("""
L’un des traits marquants de notre époque réside dans la recherche d’enrichissement collectif ou de croissance économique. **Plongeant ses racines dans les aspirations des Lumières, 
la recherche de croissance économique est étroitement associée dans nos esprits à la recherche d’une forme de bonheur ou de satisfaction collective.**  


Autrement dit, nous avons tendance à penser que la richesse est l’ingrédient nécessaire et suffisant de la satisfaction individuelle et collective. 
L’ancrage de cette conviction explique sans doute en grande partie la difficulté, voire la répugnance, de nos sociétés à envisager une forme de vie collective qui ne soit pas fondée sur la croissance économique.    

Pourtant, les **enjeux climatiques** et plus généralement écologiques pourraient y inciter. En effet, l’humanité dans son ensemble est confrontée à la nécessité de réduire son empreinte écologique globale à un rythme soutenu si elle veut pouvoir garantir la capacité des générations futures à vivre heureuses. Or, jusqu’à ce jour, **l’empreinte écologique d’un pays est fortement corrélée à son niveau de richesse.**  

Nul ne sait véritablement si la persistance à long terme de la croissance économique est compatible avec la réduction rapide de l’empreinte écologique de l’humanité. Mais que ce soit le cas ou non, qu’un découplage entre croissance économique et empreinte écologique soit possible ou pas, interroger la conviction profonde évoquée ci-dessus a toute sa pertinence.  

**Si la croissance peut se poursuivre indéfiniment comme le soutiennent certains, alors, une meilleure compréhension des déterminants du bonheur humain constituerait certainement une aide précieuse pour les décideurs. Il en irait de même si un découplage n’est pas réalisable.**  

Ainsi, nos questions de départ seront les suivantes. **Notre conviction profonde est-elle fondée ?** Et si oui, jusqu’à quel point ? Ne faut-il pas lui apporter quelques nuances ? Dans quelle mesure le bonheur d’une population peut-il être prédit ou expliqué simplement en se fondant sur le niveau de richesse par habitant ?  
**N’est-il pas possible d’enrichir notre compréhension des déterminants de la satisfaction collective en élaborant un modèle enrichi d’autres variables ?**  Peut-on construire une approche plus fine plus nuancée en élargissant notre regard - c’est-à-dire en intégrant dans notre analyse une multitude de variables ?  

Dans un contexte où le principe “la croissance fait le bonheur” est remis en question, les politiques publiques actuelles pourraient avoir intérêt à **mieux saisir les ressorts du bonheur humain afin de s’y ajuster plus précisément et d’être en capacité de maximiser le niveau de bonheur moyen au sein de leur pays**.  
C’est dans le but de pouvoir commencer à répondre à ces questions très difficiles que nous menons cette investigation.
""")
  
    st.markdown("## Objectifs du projet")
     
    
    st.markdown("""
L’objectif principal du projet est d’élaborer un **modèle du niveau de satisfaction enrichi en termes de variables explicatives et performant** en termes de puissance prédictive, en nous fondant sur un large panel de données disponibles en ligne sur les sites d’institutions a priori sérieuses, comme la Banque Mondiale, Gallup, le Bureau international du travail, etc. 

Nous espérons donc pouvoir obtenir un modèle prédictif utilisant des variables explicatives variées incluant des données relatives à la santé, l’éducation, les institutions, les inégalités économiques, etc. qui soit au moins aussi performant qu’un modèle expliquant le niveau de satisfaction en recourant uniquement au niveau de vie par habitant. C’est le **premier objectif** du projet.

À partir du travail de modélisation, nous souhaiterions pouvoir identifier les points forts et points faibles de chaque pays de façon à pouvoir **nourrir l’élaboration et la correction des politiques publiques**. Pour ce faire, nous aurons donc recours à différents modèles de régression, puis nous chercherons à interpréter les résultats obtenus par le meilleur modèle à l’aide de la bibliothèque **Shap**. Notre **second objectif** sera donc de pouvoir montrer que le modèle enrichi est capable d’**éclairer les ressorts de la satisfaction de chacun des pays de notre jeu de données**. 

                """)
    
elif page == pages[1]:
    st.markdown("## Les jeux de données")
   
    with open("test markdown-5.md", "r")as f:
        text= f.read()
    st.markdown(text)

    st.markdown("""
                ## Vue d'ensemble des jeux de données
                
                Tous n'ont pas été finalement exploités pour l'exploration en profondeur ou pour la modélisation. Cependant, très large majorité d'entre eux a été retenue.
                
                1. [Capital social:](https://bti-project.org/) ce jeu de données contient une trentaine de variables ayant trait à l’économie, à la société, aux institutions économiques, politiques, éducatives, ou de sécurité sociale, à la gouvernance.
                2. [Temps libre:](https://ourworldindata.org/time-use) usage du temps dans les pays de l'OCDE. 
                3. [Pib/habitant:](https://donnees.banquemondiale.org/indicator/NY.GDP.PCAP.KD) niveau de richesse par habitant dans le monde pour l’an 2021. En dollars constant 2015.
                4. [Mortalité infantile:](https://ourworldindata.org/child-mortality) part des nouveau-nés qui décéde avant l’âge de 5 ans. 
                5. [Démographie:](https://ourworldindata.org/world-population-growth) taux d'accroissement de la population.
                6. [Insécurité alimentaire:](https://ourworldindata.org/hunger-and-undernourishment) part de la population en situation de malnutrition modérée ou extrême.
                7. [Insécurité alimentaire:](https://ourworldindata.org/grapher/coefficient-of-variation-cv-in-per-capita-caloric-intake) inégalités alimentaires internes à chaque pays
                8. [Développement:](https://ourworldindata.org/human-development-index) indice de développement humain (espérance de vie,pib/capité, éducation).
                9. [Capital social](https://ourworldindata.org/trust) niveau de confiance interpersonnelle
                10. [Éducation:](https://ourworldindata.org/global-education)n ombre d'années moyen d'études par pays.
                11. [Éducation:](https://ourworldindata.org/global-education)n ombre d'années d'études attendues 
                12. [Éducation:](https://ourworldindata.org/tertiary-education) part de la population disposant d’un diplôme d’éducation supérieure.
                13. [Travail:](https://ourworldindata.org/working-hours) nombre d'heures travaillées par an.
                14. [Économie:](https://ourworldindata.org/grapher/economic-inequality-gini-index) niveau d’Inégalités économiques mesuré par le coefficient de Gini.
                15. [Protection des droits humains:](https://ourworldindata.org/grapher/freedom-of-association)m esure de la liberté d’action des associations de la société civile et des partis politiques.
                16. [Protection des droits humains:](https://ourworldindata.org/grapher/freedom-of-expression) il capture la mesure dans laquelle les gens peuvent exprimer leurs opinions et où les médias présentent différents points de vue politiques.
                17. [Protection des droits humains:](https://ourworldindata.org/grapher/human-rights-protection) les scores reflètent la mesure dans laquelle les citoyens sont protégés contre les meurtres du gouvernement, la torture, les emprisonnements politiques, les exécutions extrajudiciaires, les massacres, et disparitions.
                18. [Protection des droits humains:](https://ourworldindata.org/grapher/civil-liberties) il saisit dans quelle mesure les lois sont transparentes et appliquées de manière prévisible, l'administration publique est impartiale, et les citoyens jouissent d'un accès à la justice, de droits de propriété garantis, de l'absence de travail forcé, de la liberté de mouvement, du droit à l'intégrité physique et de la liberté de religion.
                19. [Régime politique:](https://ourworldindata.org/democracy) classement des pays en fonction de leur régime politique (autocratie, autocratie électorale, démocratie électorale, démocratie libérale).
                """)
    
    
    st.markdown("""
                ## Les ouvrages consultés 
                
                
                Parmi des sources expertes consultées, il paraît intéressant de mentionner le livre de Claudia Senik intitulé [L’économie du bonheur](https://www.seuil.com/ouvrage/l-economie-du-bonheur-claudia-senik/9782021186239), celui de Richard Layard intitulé [Le prix du Bonheur](https://www.dunod.com/histoire-geographie-et-sciences-politiques/prix-du-bonheur-lecons-d-une-science-nouvelle),  [La Grande Évasion](https://www.puf.com/content/La_grande_évasion_0) d’Angus Deaton, et [Cultural Evolution](https://www.cambridge.org/fr/academic/subjects/politics-international-relations/comparative-politics/cultural-evolution-peoples-motivations-are-changing-and-reshaping-world?format=HB&isbn=9781108489317) de Ronald Inglehart. Claudia Senik présente une vue d’ensemble de l’économie du bonheur en partant du célèbre [paradoxe d’Easterlin](https://fr.wikipedia.org/wiki/Paradoxe_d%27Easterlin) qui fait peser un doute sur l’existence d’une relation totalement univoque entre croissance du pib par habitant et croissance du bonheur moyen des populations. 
                
                Elle expose, en particulier, la **théorie des perspectives**, développée par le prix Nobel d’économie Daniel Kahneman, qui affirme que l’évaluation du bonheur ne se fait pas en référence à un niveau absolu de richesse, mais suppose la médiation d’un référentiel de normes sociales qui peut varier dans le temps et fonction du stade de développement d’un pays. Ronald Inglehart, en prenant appui sur les enquêtes sur l’évolution des valeurs au niveau mondiale, montre, qu’au cours des cinquante dernières années, le développement économique a eu un impact important sur ce que les populations du monde valorisent en termes d’égalité, d'éducation, de politique, de liberté individuelle, etc. 
                
                Autrement dit, on peut en déduire que ce qui compte dans le niveau de bonheur d’un pays est peut-être fonction du niveau de richesse mais aussi vraisemblablement du système de valeurs. La **théorie des perspectives** suggère également que notre jugement sur la satisfaction est sensible aux variations par rapport au référentiel, et que lorsque la variation devient la nouvelle norme son impact sur la satisfaction tend à s’évanouir (ce qui est baptisé _tapis roulant hédoniste_ ou _habituation_). 
                
                L’idée de référentiel servant de base au jugement de satisfaction renverrait selon les économistes à notre **rapport aux autres**. En effet, la satisfaction que nous éprouvons d’avoir un certain niveau de vie serait relative à la situation de personnes avec qui nous nous pensons comparables (famille, voisin, camarade de classe, etc.). Ce qui suggère que la question des inégalités au sein d’une population pourrait avoir une influence sur le niveau de satisfaction, toutes choses étant égales par ailleurs. 
                
                Dans _La Grande Évasion_, Angus Deaton propose quelques critiques classiques à l’idée d’une relation univoque entre niveau de richesse par habitant tel que mesuré par le Pib et le niveau de satisfaction dans la vie. 
                
                **En premier lieu**, il rappelle que **la mesure le Pib ne distingue pas entre les types de valeurs créées**. Certains types de valeurs sont susceptibles de nuire au niveau de satisfaction des populations tout en augmentant le pib par habitant. La militarisation à outrance, une population en mauvaise santé mentale ou psychique, des écosystèmes dégradés peuvent générer des activités économiques venant accroître le niveau de richesse tout en faisant référence à une situation potentiellement néfaste pour les populations. 
                
                En **second lieu**, il met le doigt sur le fait que le pib ne tient pas compte ou sous-évalue l’incidence de changements qui sont susceptibles d’accroître notre bonheur quotidien. Il donne en exemple les possibilités d’occupation de notre temps libre apportées par internet, les smartphones, et les nombreuses innovations ayant eu lieu au cours des vingt dernières années et dont l’intégration dans la mesure du pib est très difficile. Deaton explore également les liens entre bonheur, santé, pauvreté, éducation, et richesse.
                
                **Pour résumer** rapidement l’influence de ces lectures, nous pourrions dire qu'elles ont donné **une justification à la problématique du projet** en ceci qu'elles ont montré qu’il y avait probablement une recherche fructueuse à mener pour mieux comprendre les déterminants du bonheur, de la satisfaction des populations, en dépit du fait que le niveau de richesse joue certainement un rôle important. 
                
                Au-delà de ce premier apport, elles ont **guidé la recherche de données complémentaires**. Par exemple, en ayant en tête l’idée que ce que nous valorisons change en fonction du développement d’un pays, nous avons cherché des données relatives à des aspects non matériels de l’existence comme la liberté d’expression, d’association, entre autres choses. 
                
                **Enfin**, elles ont **stimulé l’exploration des données**. À titre d’exemple, l’idée que nous serions sensibles aux variations nous a conduit à construire une variable donnant le taux de croissance des dix dernières années. Les travaux cités suggèrent également que nous pouvons nous attendre à ce que la modélisation montre que les variables qui comptent pour la prédiction du niveau de satisfaction changent en fonction du niveau de développement d’un pays. 

                """)
    


elif page == pages[2]:
     
   st.markdown("## Explorations des données")
   st.image("AdobeStock_291576437.jpeg",width=600)

   st.subheader("Jeu de données")
   st.write("Nous présentons ici un jeu de donnée légèrement réduit en nombre de variables afin que l'utilisateur puisse s'y retrouver plus aisément.")
   
   
   st.dataframe(df)


  # options_df = st.multiselect('Je filtre les données',df["Pays"],format_func = lambda i: df['Pays'][i])

   #st.write('Je regarde les données pour:', options_df)  
   
   #st.dataframe(df[options_df])

   st.subheader("Éléments statistiques")
   st.write("Dans le tableau ci-dessous, vous pouvez retrouver la moyenne, l'écart-type (std), la valeur minimale, maximale,la médiane (50%), ainsi que les premier et troisième quartiles de la distribution de chaque variable.")

# AFFICHAGE DES VALEURS de VARIABLES (Moyenne, st, Min, Max) 
   option_metric = st.selectbox(
    'Quelle variable voulez-vous explorer ?',options = df.columns[3:])

   col1, col2, col3, col4 = st.columns(4)
   col1.metric("moyenne", df.mean().round(1)[option_metric])
   col2.metric("Max", df[option_metric].round().max())
   col3.metric("Min", df[option_metric].round().min())
   col4.metric("Écart-type", df.std()[option_metric].round(1))


   st.write("Résumé des principales statistiques")
# Affichage du tableau résumant les principales statistiques des variables
   st.dataframe(df.describe())
   #st.metric(label=f"{option_metric} moyenne", value= df.mean().round(1)[option_metric], delta= None)




# TEST CARTOGRAPHIE 

   import geopandas as gpd
   import contextily as ctx
   import matplotlib.pyplot as plt
   import folium,mapclassify
   import pandas as pd

   import streamlit as st

   #df = pd.read_csv('df_modélisationWHR')
   #df = df.drop("Unnamed: 0", axis = 1)

# Préparation du jeu de données géopsptiale avec les données à représenter
   gpd = gpd.read_file('custom.geo-2.json')

   gpd["sovereignt"]= gpd["sovereignt"].str.upper()
   gpd = gpd.rename(columns={"sovereignt": "Pays"})
   gpd = gpd.replace(to_replace="UNITED STATES OF AMERICA", value= "UNITED STATES" )
   gpd = gpd.replace(to_replace="DEMOCRATIC REPUBLIC OF THE CONGO", value= "CONGO (BRAZZAVILLE)" )
   gpd = gpd.replace(to_replace="UNITED REPUBLIC OF TANZANIA", value= "TANZANIA" )
   gpd_merged = pd.merge(gpd, df, how='left', on="Pays")
   gpd_merged = gpd_merged.dropna(subset=['Pays'], axis=0)
   gpd_merged = gpd_merged.drop(["featurecla","scalerank", "labelrank","sov_a3"], axis=1)
   gpd_merged = gpd_merged.drop(["adm0_dif","level", "tlc","admin", "adm0_a3"], axis=1)
   gpd_merged = gpd_merged.drop(["geounit","gu_a3", "su_dif","subunit", "su_a3", "brk_diff"], axis=1)
   gpd_merged = gpd_merged.set_index("Pays")

   st.subheader("Cartographie")
   st.write("Explorer comment les différentes variables se répartissent géographiquement")
   import plotly.express as px
   import geopandas as gpd

# Création de la boîte de sélectiond es variables à représenter géographiquement
   carto_option = st.selectbox(
    'Avec quelle variable voulez-vous colorer le monde ?', options = df.columns[1:])

# Création et affichage de la carte
   fig = px.choropleth_mapbox(gpd_merged,
                           geojson=gpd_merged.geometry,
                           locations=gpd_merged.index,
                           color=carto_option,
                           center={"lat": 20.5517, "lon": -5.7073},
                           mapbox_style="open-street-map",
                           zoom=0.5)

   st.plotly_chart(fig, use_container_width=False)






   st.image("AdobeStock_260035287.jpeg", width=400)



# Histogrammes

   st.subheader("Distribution des variables")

 #if st.checkbox("Afficher les valeurs manquantes") :
  #   st.dataframe(df.isna().sum())
  
   st.write("Les hisogrammes permettent de représenter la forme de la distribution des variables.")
   
   # Bouton d'option pour les histogrammes
   option_hist_1 = st.selectbox(
   'Quelle variable voulez-vous visualiser en ordonnée ?', options = df.columns[3:])
    #option_hist_2 = st.selectbox(
   #'Comment voulez-vous grouper vos données?',options = ['Région du monde', 'Régime politique'])
   group_labels = [option_hist_1] # name of the dataset

   #hist_surv = ff.create_distplot(option_hist_1, group_labels)

# Création de l'histogramme 
   hist_surv = px.histogram(df, x = option_hist_1, nbins=20) #, color = option_hist_2,  barmode = "group")
 
   st.plotly_chart(hist_surv)
 #hist_sun = px.sunburst(df, path = ['Sex', 'Pclass', 'Survived'])
 #st.plotly_chart(hist_sun)
   #st.write("Graphique Quantiles-quantiles") 
   
   #from statsmodels.graphics.gofplots import qqplot
   #import plotly.graph_objs as go
   #import plotly.figure_factory as ff

   #qqplot_data = qqplot(df["Satisfaction"], line='s').gca().lines
   #st.plotly_chart(qqplot_data)

   
   
   
# Nuage de points 
   
   st.subheader("Visualiser les relations entre variables")
   
   st.markdown("Explorez les relations entre variables du jeu de données en choisissant les variables à faire apparaître en ordonnée, en abscisse, et en couleur. Jouer avec les variables permet de reprérer visuellement des corrélations. Vous pouvez ensuite tester votre intuition en réalisant le test de Pearson.")

   option_1 = st.selectbox(
    'Quelle variable voulez-vous visualiser en ordonnée pour le nuage de points?',options = df.columns[3:])

   option_2 = st.selectbox(
   'Quelle variable voulez-vous visualiser en abscisse pour le nuage de points',options = df.columns[3:])
   
   
   option_3 = st.selectbox(
   'Avec quelle variable voulez-vous colorer vos points?', options = df.columns[1:])
   
   fig = px.scatter(df, x = option_2, y= option_1, color = option_3, 
   hover_data=['Pays'])        
   st.plotly_chart(fig)
    
   
    
# Test statistique de Pearson    
   st.subheader("Évaluer les relations entre variable avec le Test de Pearson")


   st.write("""Les test de Pearson permet d'évaluer la force d'une corrélation, ainsi que sa significativité. 
            Le coéfficient de Pearson, compris entre -1 et 1 indique la force de la corrélation. 
            La p-value indique la probabilité que la relation entre les deux variables soit due au hasard. Plus la p-value est faible, plus faible est la probabilité que la le lien entre les variable soit du au hasard. 
            Le plus souvent, on estime que si la p-value est inférieur à 5%, le lien est significatif.""")
   st.write("___")
  
   from scipy.stats import pearsonr
  
# Suppression des valeurs manquantes    
   df_test = df.dropna()
# Bouton de sélection des variables à tester
   option_test_1 = st.selectbox(
     'Test statistique : variable 1',options = df_test.columns[3:])
   option_test_2 = st.selectbox(
     'Test statistique : variable 2',options = df_test.columns[3:])

# Réalisation du test de Pearson et création du dataframe de présentation des résultats
   test = pd.DataFrame(pearsonr(df_test[option_test_1],df_test[option_test_2]), index=['pearson_coeff','p-value'], columns=['resultat_test'])
   st.write(test)

 






elif page == pages[3]: 
    st.markdown("## Présentation de la modélisation et de ses résultats")
    
        
    st.markdown("""### Description du processus de modélisation
                """)
    
    with open("Pres_modelisation.md", "r")as f:
        text_2= f.read()
    st.markdown(text_2)
 
    st.markdown("### Modélisation univariée")
                
    st.markdown("""
                Un de nos objectifs est de bâtir un modèle au moins aussi puissant qu’un modèle fondé sur la seule variable de la richesse par habitant. Pour valider cette objectif, il convient d’entraîner un modèle ne comprenant que deux variables, à savoir l’échelle de satisfaction dans la vie et le pib par habitant mise à l’échelle logarithmique. L’entraînement des modèles s’est fait dans les mêmes conditions que pour la modélisation multivariée que nous allons présenter ci-après. Conditions décrites ci-dessus. Cependant, un nombre plus restreint de modèles, comme la liste ci-dessous le montre : 
                *Ridge  
                *Lasso   
                *ElasticNet   
                *KNeighborsRegressor   
 Les métriques d'évaluation qui ont été calculées sont celles présentées dans la section précédente. Le dataframe ci-dessous présente les résultats obtenus.
                """)
    


    st.markdown("""
                Le modèles KNN a le coefficient de détermination le plus élevé avec un score de 0,84. Et sur toutes les autres métriques, c’est lui qui obtient les meilleurs résultats. Les trois modèles linéaires obtiennent des résultats très similaires. 
Nous avons ensuite représenté la distribution des erreurs grâce à un histogramme complété par une courbe de densité, et une boîte à moustache. Les erreurs suivent à peu près une loi normale. La distribution des erreurs montre plusieurs valeurs extrêmes.
Maintenant que nous avons entraîné un modèle de référence, nous allons passer à la modalisation avec l’ensemble de nos variables afin de voir si nous pouvons construire un modèle plus performant et plus riche. 
                """)
    st.image("df_modelisation_univariee.jpg")    

    st.image("AdobeStock_303401576.jpeg", width = 500)


    st.markdown("### Modélisation multi-variée")  
    
    st.markdown("La table ci-dessous résume l’ensemble des métriques utilisées pour l’ensemble des modèles entraînés. Elle présente également les hyperparamètres retenus par la recherche sur grille." )
# Mettre image ... table
    st.image("dfmetriques.jpg")    

    #from PIL import Image
   # image = Image.open('df modélisation univariée')
    
    st.markdown("Comme le montrent le tableau ci-dessus ainsi que les graphiques en barres ci-dessous, nous pouvons constater que le modèle XGBoost obtient les meilleures performances pour toutes les métriques sauf une, l’erreur maximale, où il est devancé par l’algorithme de Random Forest. Plus généralement, ces deux modèles se détachent des autres modèles pour l’ensemble des métriques. Le modèle XGBoost explique un peu plus de 96% de la variance de notre jeu de données, légèrement devant le modèle de Random Forest dont la performance frôle les 96%.  L’entraînement avec les réglages opéré dès le départ a donc donné des résultats satisfaisants, c’est pourquoi, nous n'avons pas poussé plus avant la recherche en cherchant à optimiser le meilleur modèle.")
# Mettre image ....R2
    st.image("r2.jpg")    

    st.markdown("Comme évoqué plus haut, les graphiques ci-dessous montrent comment les deux modèles les plus performants en termes de coefficient de détermination se détachent sur les autres métriques également. Le lecteur pourra remarquer que le modèle SVR donne des résultats tout à fait médiocre pour toutes les métriques. Si le modèle XGBoost est le plus performant, nous pouvons remarquer que le Random Forest est tout à fait proche.")
# Mettre image 4 métriques  
    st.image("metriques_suite.jpg")    

    st.markdown("Afin de compléter la comparaison entre les modèles les plus performants, nous avons examiné la distribution des résidus pour les modèles XGBoost et Random Forest. Nous pouvons constater que le XGBoost présente une distribution des erreurs plus concentrée, plus symétrique, avec moins de valeurs extrêmes. ")
# Distribution des erreurs XGboost et RF
    st.image("distresidus1.jpg")    
    st.image("distresidus2.jpg")    

    st.markdown("Compte tenu des éléments de comparaison présentés, c’est le modèle XGBoost qui a été retenu pour la phase d’interprétation avec la bibliothèque Shap.")
 
    
elif page == pages[4]: 

    st.markdown("## Interprétation des prédictions du modèle")

 
# Chargement du mdoèle     
    xgb_whr = load('xgb_whr.joblib')
# Chargmeent du jeu de données
    X = pd.read_csv("X_whr.csv")
# Suppression d'une colonne inutile
    X= X.drop(['Unnamed: 0'], axis = 1)
# Réalisation des prédictions du modèle
    xgb_whr.predict(X)


# Changement du nom des colonnes 

    col={"CN": "Pays", 
     "RI": "Région du monde", 
     "RP": "Régime politique", 
     "LS":"Satisfaction",
     "LGDPc": "Pib/habitant (Log)",
    "GDPc": "Pib/habitant (2020)",
    "Gr10":"Croissance du pib/habitant sur 10 ans" ,
     "SoS": "Soutien social", 
     "HLE": "Espérance de vie en bonne santé",
     "FLC": "Liberté de faire des choix", 
     "G": "Générosité",
     "POC": "Perception de la corruption",
     'IQR': "Inégalités de satisfaction au sein d'un pays",
     'MI': "Mortalité infantile",
     "GrPOP":"Accroissement naturel de la population",
     "SCOEXP":"Scolarité attendue à l'avenir",
     "SCO3": "Taux de diplômés de l'enseignement supérieur",
     "CVCI": "Inégalités alimentaire interne à chaque pays", 
     "FCIV": "Libertés civiles",
     "FEXPR": "Liberté d'expression",
     "SCOm": "Nombre moyen d'années de scolarisation",
     "GINI": "Indice de GINI",
     "MONFO": "Monopole usage de la force",
     "EDPOL": "Politique R&D",
     "POLCOOR": "Coordination des politiques",
     "NIRD": "Non intrusion de la religion en politique",
     "REFF": "Efficience des ressources"}
    X = X.rename(columns=col)


# AJOUT de l'affichage des prédiction par pays sélectionné + la valeur réèlle pour le pays
    #pays_pred = st.selectbox(
     #'Pour quel pays voulez-vous effectuer une prédiction ?',options = "???")

    #pred =  xgb_whr.predict("Mon pays")
    #st.header(f"la prédiction pour le pays : {pays} est :")
    #st.write(pred)
    #st.write('---')


# Création des valeurs de Shapley 

    exp = shap.TreeExplainer(xgb_whr) 
    vals = exp.shap_values(X)
    #st.dataframe(vals)
    
   
    
    st.markdown("### Summary plot")
    
    st.write("""Ce graphique met en évidence l'importance globale de chaque variable dans les prédictions de l’algorithme. Il présente la valeur moyenne absolue de cette variable sur tous les échantillons donnés.
             
             """)
    summary_plot = shap.summary_plot(vals, X, plot_type = 'bar',show=False)

    #summary_plot = shap.summary_plot(vals, X)
   # st.pyplot(summary_plot, bbox_inches = 'tight')  
    st_shap(summary_plot, height=600)  

    st.write('---')
    
  
    

    st.subheader("SHAP Scatter Plot")
  
    st.image("AdobeStock_374487966.jpeg", width = 400)
    st.write("Le graphique de dépendance permet de comprendre comment une variable donnée influence les prédictions en fonction de sa valeur. Les points représentent les observations (pays). En ordonnée, sont données les valeurs de shapley, c'est-à-dire l'influence (positive ou négative) de la variable sur la prédiction de la variable cible. En abscisse, on trouve les valeurs prises par la varible dans le jeu de données. Il est également possible de colorer les points en fonction d'une autre variable.")
   # st.write("Il se lit de la façon suivante ...")

# Création de la figure
    fig, ax = plt.subplots()
 
    # Création du bouton qui permet de sélectionner des pays   
    option_dependance_plot = st.selectbox(
     'De quelle variable voulez-vous visualiser l''influence?', options = X.columns)

# Création du bouton qui permet de sélectionner des pays   
    option_dependancee_plot_2 = st.selectbox(
     'Avec quelle variable voulez-vous colorer les points ?', options = X.columns)

    st.set_option('deprecation.showPyplotGlobalUse', False)
# Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(f"Influence de la variable '{option_dependance_plot}' de choix sur les prédictions")
    shap.dependence_plot(option_dependance_plot, vals, X, interaction_index = option_dependancee_plot_2 , ax= ax,show=False)
    st.pyplot( bbox_inches = 'tight')  
    st.write('---')


    st.subheader("Interprétation locale : Force Plot")
    st.image("AdobeStock_423722898.jpeg", width=300)

    st.markdown("""Le graphique 'force_plot' de SHAP permet de clarifier l'importance des variables dans la prédiction faite par l'algorithme. 
                En rose, sont indiquées les variables qui contribuent à accroître la prédiction, tandis que les variables minimisant la prédiction sont colorées en bleu. 
                Sont indiquées en sus les valeurs prises par les variables. 
                """)
    st.write("___")
    

    X = X.set_index(df['Pays'])
    #st.dataframe(X.T)

    st.dataframe(X.head())
    #vals_2 = pd.DataFrame(vals)
    #vals_2 = vals_2.set_index(df['Pays'])
    #st.dataframe(pd.DataFrame(vals_2))

# PAYS 1

    fig, ax = plt.subplots()
  
    pays = X.T.columns
    #option_force_plot = st.selectbox('Je sélectionne :', options = range(len(pays)), format_func = lambda i: pays[i])
    option_force_plot = st.selectbox('Je sélectionne :', options = range(len(pays)), format_func = lambda i: df['Pays'][i])


    ax.set_title(f"{X.index}")
    pi = shap.force_plot(exp.expected_value, vals[option_force_plot, :], X.iloc[option_force_plot,:])#,ax= ax,show=False) 
    st_shap(pi)  
  
    
    #option_faffichage_LS = st.selectbox('Satsifaction mesurée:', options = range(len(pays)), format_func = lambda i: df['Pays'][i])

    col1, col2 = st.columns(2)
    col1.metric("Valeur réèlle", df["Satisfaction"][option_force_plot])
    col2.metric("Erreur de prédiction", (df["Satisfaction"][option_force_plot]-xgb_whr.predict(X)[option_force_plot]).round(2))
  
    st.write('---')

# PAYS 2

    fig, ax = plt.subplots()
 
    pays_2 = X.T.columns
 #option_force_plot = st.selectbox('Je sélectionne :', options = range(len(pays)), format_func = lambda i: pays[i])
    option_force_plot_2 = st.selectbox('Je compare à :', options = range(len(pays_2)), format_func = lambda i: df['Pays'][i])

    ax.set_title(f"{X.index}")
    pi = shap.force_plot(exp.expected_value, vals[option_force_plot_2, :], X.iloc[option_force_plot_2,:])#,ax= ax,show=False) 
    st_shap(pi)  
    
    #option_faffichage_LS_2 = st.selectbox('Satisfaction mesurée (second pays):', options = range(len(pays)), format_func = lambda i: df['Pays'][i])

    col1, col2 = st.columns(2)
    col1.metric("Valeur réèlle", df["Satisfaction"][option_force_plot_2])
    col2.metric("Erreur de prédiction", (df["Satisfaction"][option_force_plot_2]-xgb_whr.predict(X)[option_force_plot_2]).round(2))
  
    
    st.write('---')


    #st.dataframe(X.loc[option_force_plot])
    shap.initjs()

    st_shap(shap.force_plot(exp.expected_value, vals, X))


# MULTISELECTION DE PAYS

    #options2 = st.multiselect(
    #'What are your favorite colors',
    #range(len(pays_2)),format_func = lambda i: df['Pays'][i])

    #st.write('You selected:', options2)
    
    #for o in options2: 
     #   fig, ax = plt.subplots()
      #  shap.initjs()

       # pi2 = shap.force_plot(exp.expected_value, vals[options2, :], X.iloc[options2,:])#,ax= ax,show=False) 
        #st_shap(pi2)  
    #st.write('---')






else : 
    
    st.header("Conclusions et ouverture")
    st.markdown(""" 
        Si notre méthode d’investigation a été sans lacune, nous sommes parvenus à construire un modèle relativement performant qui permet de cerner les **“ingrédients” de la satisfaction** moyenne des populations du monde en nous fondant sur une pluralité
        de variables faisant référence à de nombreux aspects de la vie humaine.
        Si l’on s’en tient au coefficient de détermination de notre modèle de référence et du modèle XGBoost,
        nous constatons même qu’une approche multi-variées rend davantage compte de la variance présente dans notre jeu de données qu’un modèle fondé uniquement sur le niveau de vie par habitant.
        Toutefois, il serait faux d’affirmer que notre modèle **XGBoost** fait fi du niveau de richesse.   
        
        Le niveau de richesse d’un pays entretient, comme nous l’avons vu, une relation statistiquement significative avec le niveau de satisfaction moyen de ce pays.
        Notre modèle permet donc plutôt d’avoir une compréhension plus riche de la relation entre le niveau de vie et de satisfaction.
        **Il ne suffit pas d’être riche pour être heureux, encore faut-il prêter attention à d’autres aspects de l’existence humaine que notre modèle permet de mettre en évidence.**   
        
        Parvenus au terme de notre enquête, il est intéressant de porter un regard rétrospectif et critique  sur le chemin parcouru. 
        
        Si nous pouvions recommencer, que ferions-nous différemment ?  Et comment pourrions-nous enrichir notre démarche ? 
        Il serait possible de **construire plus de variables** à partir des données disponibles, notamment des variables synthétisant une dynamique temporelle, comme nous l’avons fait avec la croissance à 10 ans. 
        On pourrait ainsi construire des variables mesurant la rapidité du changement du niveau de l’espérance de vie, du nombre d’années d'éducation, de la fertilité, etc. 
        Cela permettrait de mieux apprécier le poids potentiel des variations dans les jugements de satisfaction.  
        
        **Une autre voie** consisterait à essayer de construire une **série temporelle** avec un maximum de variables afin de pouvoir intégrer d’une autre façon la dynamique temporelle. 
        Toutefois, il n’est pas certain que le travail de collecte des données serait aisé, voire tout simplement possible, sans réduire à peu de variables le jeu de données, ou sans accepter un nombre important de valeurs manquantes. 
        Une autre façon d’enrichir ce travail consisterait à utiliser **DICE-ML** pour interpréter le modèle retenu et pour concevoir des expériences d’intervention.""")  
        
    st.image("AdobeStock_487090715.jpeg", width=700)
   
        
    st.markdown("""Enfin, il serait très intéressant dans le contexte actuel de tenter d’**élargir le problème initial** pour qu’il réponde mieux aux enjeux des gouvernements confrontés à l’exigence de **transition écologique**. 
        Supposons qu'un gouvernement veuille réduire son empreinte écologique sans réduire son niveau de bonheur, ou en le réduisant le moins possible ? 
        **Dans quels aspects de notre vie collective doit-il mener des actions ?** Faut-il mettre l’accent sur la santé, l’éducation, la réduction des inégalités, le contrôle des naissances, la promotion de la solidarité familiale et amicale, la liberté d’expression, la recherche scientifique et technique, etc. ?""") 
        
    st.markdown("**Notre étude a-t-elle quelque chose à dire à ce sujet ?** Peut-elle aider ? Pas vraiment, pas directement, même si elle permet de comprendre qu’il y a sans doute des chemins vers une plus grande satisfaction qui passent par un certain recul par rapport à une recherche effrénée de croissance du pib/habitant.") 
   
     
    st.image("AdobeStock_260037651.jpeg", width=700)
    
