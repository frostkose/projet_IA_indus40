# projet-IA-indus4.0
 Prédiction des Arrêts de  Protection d’un Cobot fait par Bouzidi Safa &amp; Khalfa Youssef





### 🧠 **Méthodologie Adoptée**

Le projet consiste à prédire les défaillances d’équipements industriels à partir de données issues de capteurs. Voici les étapes suivies :

---

#### 🔧 **Prétraitement des données**
- Suppression des valeurs manquantes  
- Normalisation des features  
- Construction de séquences temporelles (fenêtre glissante de 10 pas de temps)  

#### 🗂️ **Préparation du dataset**
- Création des labels binaires (0 : pas de panne, 1 : panne)  
- Séparation en ensembles d’entraînement, de validation et de test  

#### 🧪 **Développement du modèle**
- Modèles LSTM pour la détection des anomalies ou des pannes  
- Architecture adaptée aux séries temporelles multivariées  

#### 📊 **Entraînement et évaluation**
- Grid search sur les hyperparamètres  



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


🚀 Instructions pour exécuter l’API et utiliser le modèle
---

📦 Installation des dépendances
---

pip install -r requirements.txt

---

▶️ Lancement de l’API (FastAPI)
---

uvicorn main:app --reload
---

