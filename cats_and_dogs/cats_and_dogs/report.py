import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Função para salvar as métricas no arquivo de log
def save_metrics(metrics_data, experiment_name):
    log_path = os.path.join(LOG_DIR, f"{experiment_name}.json")

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)
        logs.append(metrics_data)
    else:
        logs = [metrics_data]

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=4)

# Função para obter logs do experimento
def get_experiment_logs():
    logs_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.json')]

    if not logs_files:
        print("Nenhum log encontrado.")
        return None, None

    print("Experimentos disponíveis:")
    for idx, file in enumerate(logs_files, 1):
        print(f"{idx}. {file}")
    
    try:
        experiment_idx = int(input("\nEscolha o número do experimento para visualizar: ")) - 1
        if experiment_idx < 0 or experiment_idx >= len(logs_files):
            print("Número inválido. Tente novamente.")
            return None, None
    except ValueError:
        print("Entrada inválida. Tente novamente.")
        return None, None

    experiment_name = logs_files[experiment_idx].split('.')[0]
    log_path = os.path.join(LOG_DIR, logs_files[experiment_idx])

    with open(log_path, "r") as f:
        logs = json.load(f)

    return experiment_name, logs

# Função para extrair as métricas do log
def extract_metrics_from_logs(logs):
    epochs = [entry["epoch"] for entry in logs]
    train_loss = [entry["train_loss"] for entry in logs]
    val_loss = [entry["val_loss"] for entry in logs]
    train_acc = [entry["train_acc"] for entry in logs]
    val_acc = [entry["val_acc"] for entry in logs]

    # Pegando y_true e y_pred do último epoch disponível
    y_true = logs[-1].get("y_true_val", [])
    y_pred = logs[-1].get("y_pred_val", [])

    return epochs, train_loss, val_loss, train_acc, val_acc, y_true, y_pred

# Função para plotar as métricas do experimento
def plot_metrics():
    experiment_name, logs = get_experiment_logs()

    if logs is None:
        return

    epochs, train_loss, val_loss, train_acc, val_acc, y_true, y_pred = extract_metrics_from_logs(logs)

    if not y_true or not y_pred:
        print("Aviso: y_true e y_pred não encontrados no log. Matriz de confusão não será gerada.")
        return

    # Criar figura
    plt.figure(figsize=(14, 10))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', color='red')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Validation Loss por Epoch")
    plt.xticks(epochs)
    plt.legend()

    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy', marker='o', color='green')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train & Validation Accuracy por Epoch")
    plt.xticks(epochs)
    plt.legend()

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.subplot(2, 2, 3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Métricas de performance
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = [f1, precision, recall, accuracy]
    metrics_names = ['F1-Score', 'Precision', 'Recall', 'Accuracy']

    plt.subplot(2, 2, 4)
    plt.bar(metrics_names, metrics, color=['blue', 'green', 'orange', 'red'])
    plt.title("Performance Metrics")
    plt.ylabel("Score")
    plt.ylim([0, 1])
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f"{experiment_name}_full_metrics.png"))
    plt.show()

# Execução principal
if __name__ == "__main__":
    plot_metrics()
