import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # converte classe para numerico e normaliza
    data['Class'] = data['Class'].map({'Kecimen': 0, 'Besni': 1})

    # Separa atributos (X) e classe (y)
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Normaliza os atributos
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Divide em treino, validacao e teste
    X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    # Salva os dados em arquivos CSV
    pd.DataFrame(X_train).assign(Class=y_train.values).to_csv('data/train.csv', index=False)
    pd.DataFrame(X_val).assign(Class=y_val.values).to_csv('data/val.csv', index=False)
    pd.DataFrame(X_test).assign(Class=y_test.values).to_csv('data/test.csv', index=False)

if __name__ == "__main__":
    file_path = "data/Raisin_Dataset.csv"
    data = load_data(file_path)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(data)

    save_data(X_train, X_val, X_test, y_train, y_val, y_test)
    print("Dados pre-processados e salvos!")
