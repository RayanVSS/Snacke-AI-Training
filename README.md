# Snake AI Training

Ce projet propose une implémentation d'un jeu Snake accompagné d'une intelligence artificielle (IA) qui s'entraîne à jouer grâce à un algorithme de Q-learning. L'architecture sépare la logique du jeu (l'environnement) de l'agent d'IA, permettant ainsi d'entraîner l'agent en mode non graphique et de lancer une démonstration graphique pour observer ses performances.


## Sommaire

1. [Fonctionnalités](#fonctionnalités)
2. [Prérequis](#prérequis)
3. [Utilisation](#utilisation)
    - [Entraînement de l'IA](#entraînement-de-lia)
    - [Démonstration Graphique](#démonstration-graphique)
4. [Architecture du Projet](#architecture-du-projet)
5. [Améliorations Futures](#améliorations-futures)
6. [Statistique](#statistique)

## Fonctionnalités

- **Jeu Snake avec Pygame :**
    
    Un environnement complet de jeu Snake avec gestion des collisions, déplacement du serpent et affichage graphique.
    
- **Agent d'IA par Q-learning :**
    
    Un agent qui apprend à jouer en mettant à jour des Q-valeurs selon les récompenses reçues.
    
- **Entraînement multi-threadé :**
    
    Utilisation du multi‑threading pour accélérer l'entraînement en lançant plusieurs épisodes en parallèle.
    
- **Sauvegarde continue des données :**
    - Les Q-valeurs sont enregistrées dans le fichier `q_values.pkl` pour poursuivre l'entraînement ultérieur.
    - Les scores de chaque épisode sont sauvegardés dans le fichier `training_data.csv` avec une numérotation continue des épisodes.
- **Lancement en ligne de commande :**
    
    Possibilité de spécifier le nombre d'épisodes d'entraînement directement via la ligne de commande.
    

## Prérequis

- **Python 3.6+**
- **Bibliothèques Python :**
    - `pygame`
    - `numpy`
    - Les modules standard `argparse`, `pickle`, `csv`, `threading`, `os`, `random`, `sys`

Pour installer les dépendances, vous pouvez utiliser pip :

```bash
pip install pygame numpy
```

## Utilisation

### Entraînement de l'IA

Pour entraîner l'IA, lancez le script en mode `train` en spécifiant le nombre d'épisodes souhaité. Par exemple, pour entraîner l'agent pendant 2000 épisodes :

```bash
python snack_IA.py train 2000
```

> Note :
> 
> - Si le fichier `training_data.csv` existe déjà, les nouveaux épisodes seront ajoutés à la suite avec une numérotation continue (ex. : si le dernier épisode enregistré est le 5000, le prochain sera le 5001).
> - Les Q‑valeurs sont sauvegardées dans `q_values.pkl` pour permettre de reprendre l'entraînement sans repartir de zéro.

### Démonstration Graphique

Pour observer l'IA jouer en mode graphique, lancez :

```bash
python snack_IA.py demo
```

Assurez-vous d'avoir déjà entraîné l'IA (au moins quelques épisodes) afin que le fichier `q_values.pkl` soit disponible.

## Architecture du Projet

- **SnakeGame :**
    
    Gère la logique du jeu, l'affichage avec Pygame, la gestion des collisions, et fournit l'état du jeu pour l'agent.
    
- **Agent :**
    
    Implémente le Q-learning pour apprendre à jouer. L'agent met à jour ses Q‑valeurs en fonction des actions et des récompenses obtenues.
    
- **Entraînement multi-threadé :**
    
    L'entraînement est réparti sur plusieurs threads pour accélérer le processus. Les accès concurrents aux Q‑valeurs et à l'écriture dans le CSV sont protégés par des verrous.
    
- **Sauvegarde des données :**
    
    Les scores de chaque épisode et le numéro d'épisode sont enregistrés dans `training_data.csv` pour permettre une analyse ultérieure des performances (par exemple, pour générer des graphiques).
    

## Améliorations Futures

- Intégrer le Deep Q-Learning (DQN) pour des performances améliorées.
- Optimiser la représentation de l'état et la fonction de récompense.
- Développer des visualisations avancées des performances de l'agent (graphiques, statistiques).
- Explorer le multi-processing pour contourner les limitations du GIL en Python.

## Statistique