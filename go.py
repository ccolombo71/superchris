import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# Leggo il file Excel
df = pd.read_excel('sestine.xlsx', header=None)
df.columns = [f'num_{i+1}' for i in range(6)]

# Converto il DataFrame in array numpy per l'elaborazione
data = df.values

# Funzione per calcolare features aggiuntive
def extract_features(sestina):
    features = []
    features.extend([sestina[i+1] - sestina[i] for i in range(len(sestina)-1)])
    features.append(np.mean(sestina))
    features.append(np.std(sestina))
    features.append(max(sestina) - min(sestina))
    return features

# Aggiungo features al DataFrame
feature_names = [f'diff_{i+1}' for i in range(5)] + ['mean', 'std', 'range']
additional_features = np.array([extract_features(row) for row in data])
for i, name in enumerate(feature_names):
    df[name] = additional_features[:, i]

# Preparo i dati per il training
X = df[feature_names].values
y = df[[f'num_{i+1}' for i in range(6)]].values

# Split dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definizione modelli
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'SVM': SVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Training e valutazione
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    # Training su tutti i numeri della sestina
    predictions = np.zeros_like(y_test)
    for i in range(6):
        model.fit(X_train_scaled, y_train[:, i])
        predictions[:, i] = model.predict(X_test_scaled)
    
    # Calcolo accuracy considerando una predizione corretta se il numero è nell'intervallo ±2
    accuracy = np.mean([np.abs(predictions[j] - y_test[j]) <= 2 for j in range(len(y_test))])
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy*100:.2f}%")

# Seleziono il modello migliore
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_accuracy = results[best_model_name]
print(f"\nMiglior modello: {best_model_name} con accuracy {best_accuracy*100:.2f}%")

# Funzione per generare nuove sestine valide
def generate_valid_sestines(model, scaler, n_sestine=3, max_attempts=1000):
    valid_sestines = []
    attempts = 0
    
    while len(valid_sestines) < n_sestine and attempts < max_attempts:
        attempts += 1
        
        try:
            # Genero features casuali basate sui range osservati
            random_features = np.random.uniform(
                low=X.min(axis=0),
                high=X.max(axis=0),
                size=(1, X.shape[1])
            )
            
            # Scalo le features
            scaled_features = scaler.transform(random_features)
            
            # Predico i numeri
            predictions = []
            for i in range(6):
                pred = model.predict(scaled_features)[0]
                predictions.append(int(round(max(1, min(90, pred)))))
            
            # Ordino i numeri e rimuovo duplicati
            sestina = sorted(list(set(predictions)))
            
            # Se ho meno di 6 numeri, aggiungo numeri casuali mancanti
            while len(sestina) < 6:
                new_num = np.random.randint(1, 91)
                if new_num not in sestina:
                    sestina.append(new_num)
            
            # Ordino la sestina finale
            sestina = sorted(sestina)
            
            # Verifico che la sestina sia valida
            if (len(sestina) == 6 and 
                all(1 <= x <= 90 for x in sestina) and 
                all(sestina[i] < sestina[i+1] for i in range(5))):
                valid_sestines.append(sestina)
                print(f"Trovata sestina valida dopo {attempts} tentativi")
        
        except Exception as e:
            print(f"Errore durante la generazione: {e}")
            continue
    
    if len(valid_sestines) < n_sestine:
        print(f"\nATTENZIONE: Sono riuscito a generare solo {len(valid_sestines)} sestine valide su {n_sestine} richieste dopo {max_attempts} tentativi")
    
    return valid_sestines

# Genero e visualizzo le nuove sestine
print("\n=== SESTINE PREDETTE ===")
new_sestines = generate_valid_sestines(best_model, scaler)
if new_sestines:  # Verifico che ci siano sestine generate
    for i, sestina in enumerate(new_sestines, 1):
        print(f"Sestina {i}: {' - '.join(map(str, sestina))}")
else:
    print("Non è stato possibile generare sestine valide")
print("=====================")
