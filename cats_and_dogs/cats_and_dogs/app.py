import io
import os
import torch
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image

from cats_and_dogs.model import CNN

# Inicializa o FastAPI
app = FastAPI()

# Caminho do diretório onde os modelos são salvos
TRAIN_DIR = "train"

# Constantes
CLASS_NAMES = ['Cat', 'Dog']
CONFIDENCE_THRESHOLD = 0.8

# Pipeline de pré-processamento
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensiona para 128x128
    transforms.ToTensor(),  # Converte para tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normaliza
])

# Lista os modelos .pth disponíveis
def list_models():
    model_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.pth')]
    
    if not model_files:
        print("Nenhum modelo encontrado.")
        return None
    
    print("Modelos disponíveis:")
    for idx, file in enumerate(model_files, 1):
        print(f"{idx}. {file}")
    
    try:
        model_idx = int(input("Escolha o número do modelo: ")) - 1
        if model_idx < 0 or model_idx >= len(model_files):
            print("Número inválido.")
            return None
        return model_files[model_idx]
    except ValueError:
        print("Entrada inválida.")
        return None

# Carrega o modelo selecionado
def load_model(model_filename: str):
    model = CNN()
    model_path = os.path.join(TRAIN_DIR, model_filename)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  
    model.eval()
    return model

# Pré-processa a imagem recebida
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(img).unsqueeze(0)  # Adiciona dimensão extra

# Realiza a predição da imagem
def predict_class(model: torch.nn.Module, image: torch.Tensor) -> dict:
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(probabilities, 1).item()
        confidence = probabilities[0, predicted_class_idx].item()

    if confidence < CONFIDENCE_THRESHOLD:
        return {"predicted_class": "Unknown", "likelihood": round(confidence, 4)}

    return {"predicted_class": CLASS_NAMES[predicted_class_idx], "likelihood": round(confidence, 4)}

# Escolhe e carrega o modelo na inicialização do servidor
model_filename = list_models()  # Aqui é chamado apenas uma vez ao iniciar o servidor
if model_filename:
    print(f"Modelo selecionado: {model_filename}")
    model = load_model(model_filename)  # Carrega o modelo escolhido
else:
    print("Nenhum modelo selecionado.")
    model = None

# Endpoint para predição
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Verifica se o modelo foi carregado
    if model is None:
        return {"error": "Nenhum modelo carregado."}

    # Lê os bytes da imagem e faz a predição
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    prediction = predict_class(model, image)
    return prediction
