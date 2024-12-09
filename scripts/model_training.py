import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import os

def load_preprocessed_data():
    train_data = pd.read_csv('data/train.csv')
    val_data = pd.read_csv('data/val.csv')
    test_data = pd.read_csv('data/test.csv')

    # Atributos 'X' e rótulos 'y'
    X_train = train_data.drop(columns=['Class']).values
    y_train = train_data['Class'].values
    X_val = val_data.drop(columns=['Class']).values
    y_val = val_data['Class'].values
    X_test = test_data.drop(columns=['Class']).values
    y_test = test_data['Class'].values

    return X_train, y_train, X_val, y_val, X_test, y_test

def build_mlp_model(input_dim):
    # Definiçao da arquitetura do modelo MLP
    # Primeira camada com 14 neurônios
    # Segunda camada com 7 neurônios
    # Camada de saida com 1 neurônio
    model = Sequential([
        Dense(14, activation='relu', input_dim=input_dim),  # 2x num atributos
        Dense(7, activation='relu'),  # Reduçao progressiva
        Dense(1, activation='sigmoid')  # Saida binaria
    ])

    # Compilação do modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Taxa de aprendizado padrao
        loss='binary_crossentropy',  # Perda para classificaçao binaria
        metrics=['accuracy']  # Metrica de avaliaçao
    )

    return model

def train_and_save_model(model, X_train, y_train, X_val, y_val):
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Configuraçao de treinamento
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Num de epocas
        batch_size=32,  # Tamanho do lote padrao
        verbose=1
    )

    model.save('models/mlp_model.h5')
    print("Modelo salvo em 'models/mlp_model.h5'")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv('results/training_history.csv', index=False)
    print("Histórico de treinamento salvo em 'results/training_history.csv'")

    return history

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Desempenho no conjunto de teste: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    # Previsoes e relatório de classificaçao
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nRelatório de classificação:\n")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data()

    input_dim = X_train.shape[1]
    model = build_mlp_model(input_dim)

    train_and_save_model(model, X_train, y_train, X_val, y_val)

    evaluate_model(model, X_test, y_test)
