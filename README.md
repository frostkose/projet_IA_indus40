# projet-IA-indus4.0
 Prédiction des Arrêts de  Protection d’un Cobot fait par Bouzidi Safa &amp; Khalfa Youssef


<br>

<span style="color:blue; font-weight:bold; font-size:20px;">Méthodologie Adoptée</span>  
Le projet consiste à prédire les défaillances d'équipements industriels à partir de données issues de capteurs. Voici les étapes suivies :

<br>



<span style="color:red; font-weight:bold; font-size:16px;">Prétraitement des données :</span>  
- Suppression des valeurs manquantes  
- Normalisation des features  
- Construction de séquences temporelles (fenêtre glissante de 10 pas de temps) 

<br>

<span style="color:red; font-weight:bold; font-size:16px;">Préparation du dataset :</span>  
- Création des labels binaires (0 : pas de panne, 1 : panne)  
- Séparation en ensembles d’entraînement, de validation et de test  

<br>

<span style="color:red; font-weight:bold; font-size:16px;">Développement du modèle :</span>  
- Modèles LSTM pour la détection des anomalies ou des pannes  
- Architecture adaptée aux séries temporelles multivariées  


<br>

<span style="color:red; font-weight:bold; font-size:16px;">Entraînement et évaluation :</span>  
- Grid search sur les hyperparamètres  
- Évaluation via des métriques classiques : précision, rappel, F1-score

Comparaison des modèles et des hyperparamètres

| Modèle                        | Type               | Unités / Paramètres                | Épochs | Batch Size |
|------------------------------|--------------------|------------------------------------|--------|------------|
| **LSTM 1**                   | LSTM classique     | 2 couches LSTM (50), softmax       | 10     | 32         |
| **LSTM 2**                   | LSTM classique     | 2 couches LSTM (50), sigmoid       | 10     | 32         | 
| **LSTM 3**                   | Bi-LSTM + Dropout  | 2x Bi-LSTM (64) + Dropout(0.2)     | 15     | 64         | 
| **CNN 1D**                   | CNN                | Conv1D(64/32) + Dense(100)         | 15     | 64         | 
| **Arbre de Décision**        | Classique          | GridSearchCV sur profondeur, split | –      | –          | 
| **Random Forest**            | Classique          | 100 arbres                         | –      | –          | 
| **Régression Logistique**    | Classique          | max_iter=1000                      | –      | –          | 