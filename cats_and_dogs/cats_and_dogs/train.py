# Importações
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from report import save_metrics
from model import CNN
from settings import get_path_to_data

# Constantes
LOG_DIR = "logs"
TRAIN_DIR = "train"
EPOCHS = 20
BATCH_SIZE = 320

# Função para configurar diretórios
def setup_directories():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)

# Função para obter o nome do experimento
def get_experiment_name():
    experiment_name = input("Digite o nome do experimento (sem espaços): ")
    return experiment_name

# Função para configurar o dataset
def setup_data():
    data_dir = get_path_to_data()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# Função para configurar o modelo e otimização
def setup_model(device):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer

# Função para treinar o modelo por época
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    y_true_train, y_pred_train = [], []
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        y_true_train.extend(labels.cpu().tolist())
        y_pred_train.extend(predicted.cpu().tolist())

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    
    return train_loss, train_accuracy, y_true_train, y_pred_train

# Função para validar o modelo após cada época
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    y_true_val, y_pred_val = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            y_true_val.extend(labels.cpu().tolist())
            y_pred_val.extend(predicted.cpu().tolist())

    val_loss /= len(val_loader)
    val_accuracy = correct_val / total_val
    
    return val_loss, val_accuracy, y_true_val, y_pred_val

# Função para salvar o melhor modelo
def save_best_model(model, model_save_path, train_loss, best_loss, epoch):
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Novo melhor modelo salvo na época {epoch+1} com perda: {best_loss:.4f}")
    return best_loss

# Função principal de treinamento
def train_model():
    # Configurações iniciais
    setup_directories()
    train_loader, val_loader = setup_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    model, criterion, optimizer = setup_model(device)
    experiment_name = get_experiment_name()
    model_save_path = os.path.join(TRAIN_DIR, f"{experiment_name}.pth")
    
    best_loss = float("inf")
    
    # Loop de épocas
    for epoch in range(EPOCHS):
        # Treinamento
        train_loss, train_accuracy, y_true_train, y_pred_train = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validação
        val_loss, val_accuracy, y_true_val, y_pred_val = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Exibição dos resultados
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Salvando métricas
        metrics_data = {
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_accuracy,
            "val_acc": val_accuracy,
            "y_true_val": y_true_val,
            "y_pred_val": y_pred_val
        }
        save_metrics(metrics_data, experiment_name)
        
        # Verificação e salvamento do melhor modelo
        best_loss = save_best_model(model, model_save_path, train_loss, best_loss, epoch)

    print("Treinamento concluído!")

# Execução do treinamento
if __name__ == "__main__":
    train_model()
