
Le jeu de données initial est issu du projet *World Happiness Report 2021*. Il s’agit d’un rapport sur le bonheur mondial publié chaque année depuis la fin des années 2000 et dirigé par une équipe d’économistes de renommée internationale, spécialistes de l’économie du bonheur. La réalisation du projet nous a conduit à consulter plusieurs ouvrages issus de différents champs disciplinaires afin de nourrir l’élaboration des hypothèses de travail, la recherche de jeux de données complémentaires, ainsi que les objectifs poursuivis dans ce projet. Le nombre de variables relativement restreint contenu dans le jeu de données initial, et la volonté de pouvoir élargir nos perspectives d'analyse, nous ont conduit à chercher des sources de données complémentaires, comme nous le verrons. 

Le jeu de données initial est composé d’un échantillon de 149 pays, sur 195 reconnus dans le monde. Près d’un quart des pays existants manque donc à l’échantillon, ce qui induit la prise en compte d’une marge d’erreur de 4 à 5% selon le degré de confiance paramétré. Observons la répartition géographique de notre échantillon afin de repérer un éventuel *biais d’échantillonnage*. La répartition géographique de ces pays est la suivante :

- Sub-Saharan Africa                    36
- Western Europe                        21
- Latin America and Caribbean           20
- Middle East and North Africa          17
- Central and Eastern Europe            17
- Commonwealth of Independent States    12
- Southeast Asia                         9
- South Asia                             7
- East Asia                              6
- North America and ANZ        4     

Si nous comparons cette liste à celle des pays dans le monde, nous voyons que nous manquons principalement des données pour l’Amérique Latine et les Caraïbes (13 pays manquants), et pour l’Asie (24 pays manquants). Les pays manquants sont des pays majoritairement classés parmi les pays à bas revenus ou à revenu moyen supérieur dans la classification de la Banque Mondiale, ce qui induit probablement un **biais d’échantillonnage** limitant la portée de nos futurs résultats et conclusions. 


part, le jeu de données issu du rapport mondial sur le bonheur ne contenait aucune valeur manquante. Il présentait vingt variables au total (dont deux variables géographiques : nom de pays, et appartenance régionale). Parmi ces variables, on trouve la variable “Ladder score” qui donne le niveau de satisfaction moyen au sein d’un pays et que nous avons considérée comme notre variable cible. Nous trouvons ensuite des variables relatives à la distribution de la variable cible, précisant l’écart-type, le premier quartile, et le troisième quartile. Les autres variables explicatives, au nombre de cinq, sont 

1. **“Logged GDP per capita” :** Pib / habitant exprimé en parité de pouvoir d'achat et sur une échelle logarithmique. 
1. **“Social support” :** Pourcentage de personnes au sein d'un échantillon nationale qui déclarent pouvoir trouver du soutien en cas de problème.
1. **“Healthy life expectancy” :** Espérance de vie en bonne santé : espérance de vie sans incapacité.
1. **“Freedom to make life choices” :** les habitants sont satisfaits (ou non) de leur liberté de choix, dans leur vie personnelle.
1. **“Generosity” :** proportion des habitants qui ont fait un don à un organisme de bienfaisance, le mois dernier.
1. **“Perceptions of corruption” :** les habitants ont confiance en leur gouvernement (absence de corruption) et ont confiance dans le monde des affaires.
1. **Ladder score in Dystopia :** constante fictive qui correspond au score du pire pays imaginable. Elle est construite à partir des scores les plus bas pour chacune des variables ci-dessus. 

Le reste des variables présente la part de chaque variable dans le score de la variable cible. Nous avons donc choisi de ne conserver que les variables présentées ci-dessus, à l’exception de la dernière qui est une constante et ne présente donc aucune variance. 

Afin d’identifier des variables prometteuses susceptibles de nous aider à construire un modèle plus riche de compréhension des déterminants du bonheur, nous avons cherché d’autres jeux de données. La recherche en question a permis de multiplier par dix, environ, le nombre de variables. Les principaux sites utilisés pour la récupération des données sont les suivantes : 

- [**World Happiness Report](https://worldhappiness.report/) **:** 
  - Il s’agit du site qui présente les travaux d’une équipe internationale de chercheurs qui rédige le rapport mondial sur le bonheur depuis une dizaine d’années. Il nous a été utile pour mieux comprendre les variables contenues dans le jeu de données initial.
- [**Our World in Data](https://ourworldindata.org/) **:** 
  - Ce site regroupe une multitude de jeux de données sur plusieurs dizaines de thématiques : santé, éducation, politique, économie, bonheur. Les jeux de données sont accompagnés de dossiers thématiques rédigés par des experts et de nombreuses visualisations interactives. 
- [**Banque Mondiale](https://www.worldbank.org/fr/who-we-are/news/coronavirus-covid19) **:**
  - La Banque Mondiale est une institution financière internationale qui finance des projets de développement et met à disposition des internautes une base de données comprenant des milliers de jeux de données régulièrement mis à jour. 
- [**BTI](https://bti-project.org/en/?&cb=00000) **:** 
  - Le site du Bureau international du travail réalise une grande enquête mondiale pour analyser les processus de transformation vers la démocratie et une économie de marché en comparaison internationale et identifie les stratégies pour un changement pacifique. Ils réunissent de nombreuses données. 
- [**Global Network Footprint](https://www.footprintnetwork.org/) **:**
  - Cette organisation collecte les données relatives à l’empreinte écologique de chaque pays du monde. Elle met à disposition des internautes les jeux de données ainsi que des modules d’exploration des données. Après avoir demandé et obtenu un accès à la base de donnée du GNF, nous avons finalement renoncé à exploiter ces données qui nous auraient conduit à modifier en profondeur la problématique préalablement définie.
- [**CEPREMAP :**](https://www.cepremap.fr/)
  - Le Cepremap est l’observatoire du bien être en France qui réunit des chercheurs en sciences humaines, et en particulier des experts d’économie du bonheur comme Claudia Senik. Il met à disposition des internautes un baromètre présentant une vingtaine de variables relatives au bien être en France. Les données sont également présentées à plat et de façon croisée. Les données sont actualisées tous les mois en collaboration avec L’INSEE. 

Nous nous sommes efforcés de trouver des données collectées et compilées par des institutions de réputation internationale. Par ailleurs, notre intérêt s’est porté sur un large éventail d’options et de sujets. Ainsi les données ajoutées concernent de nombreux sujets (voir ci-dessous). Finalement, ce sont plus d’une quinzaine de jeux de données qui ont pu être fusionnés à notre jeu de données initial. Les liens vers les jeux de données ainsi que la signification des variables est renseigné dans le chapitre 1 de l’annexe. Ce faisant, le nombre de variables explicatives exploitables est passé de 6 à près de 70.  









