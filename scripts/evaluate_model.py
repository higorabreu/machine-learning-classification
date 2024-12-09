import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from tensorflow.keras.models import load_model

def load_test_data():
    test_data = pd.read_csv('data/test.csv')
    X_test = test_data.drop(columns=['Class']).values
    y_test = test_data['Class'].values
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    # Avalia o modelo e gera um relatorio de classificacao
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Desempenho no teste: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    # threshold padrao (0.5)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nRelatorio de classificacao:\n")
    print(classification_report(y_test, y_pred))

    return y_test, y_pred

def plot_confusion_matrix(y_test, y_pred):
    # matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Classe 0', 'Classe 1'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    plt.title('Matriz de Confusao', fontsize=14)
    plt.xlabel('Classe Prevista', fontsize=12)
    plt.ylabel('Classe Real', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.show()

def plot_training_history():
    # graficos de acuracia e perda ao longo das epocas
    history = pd.read_csv('results/training_history.csv')
    
    # acuracia
    plt.figure(figsize=(8, 5))
    plt.plot(history['accuracy'], label='Treinamento', linestyle='-', marker='o')
    plt.plot(history['val_accuracy'], label='Validacao', linestyle='--', marker='s')
    plt.title('Acuracia ao Longo das Epocas', fontsize=14)
    plt.xlabel('Epocas', fontsize=12)
    plt.ylabel('Acuracia', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/accuracy_plot.png')
    plt.show()

    # perda
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], label='Treinamento', linestyle='-', marker='o')
    plt.plot(history['val_loss'], label='Validacao', linestyle='--', marker='s')
    plt.title('Perda ao Longo das Epocas', fontsize=14)
    plt.xlabel('Epocas', fontsize=12)
    plt.ylabel('Perda', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/loss_plot.png')
    plt.show()

def plot_metrics_by_class(y_test, y_pred):
    # Precision, Recall e F1-Score por classe
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])

    classes = ['Classe 0', 'Classe 1']
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width=width, label='Precisao', color='skyblue')
    plt.bar(x, recall, width=width, label='Recall', color='orange')
    plt.bar(x + width, f1, width=width, label='F1-Score', color='green')

    plt.title('Metricas por Classe', fontsize=14)
    plt.xticks(x, classes, fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/metrics_by_class.png')
    plt.show()

if __name__ == "__main__":
    model = load_model('models/mlp_model.h5')

    X_test, y_test = load_test_data()

    y_test, y_pred = evaluate_model(model, X_test, y_test)

    plot_confusion_matrix(y_test, y_pred)
    plot_training_history()
    plot_metrics_by_class(y_test, y_pred)
