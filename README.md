# 🖼️ Project2: Deep Learning with PyTorch for Image Classification (CNN)

---

## 🌍 Language Options | Opções de Idioma
This README is available in two languages:
- 🇺🇸 **English (Primary)**
- 🇧🇷 **Português (Secondary)**

---

## 🇺🇸 English Version

## 📌 Project Overview
This project implements a **complete Deep Learning pipeline** for **image classification** using **PyTorch**.

The main objective is to design, train, evaluate, and deploy a **Convolutional Neural Network (CNN)** — named **ConvNet** — to classify images from the **CIFAR-10 dataset**, which contains **10 image classes** such as *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck*.

📂 **Dataset Source:**  
https://www.cs.toronto.edu/~kriz/cifar.html

---

## 🧠 Problem Context
Image classification is a core problem in **Computer Vision** and a foundational application of **Artificial Intelligence**.  
This project aims to demonstrate, in practice, how deep learning models learn visual patterns and make predictions from raw pixel data.

---

## 🎯 Project Objectives
- Build a CNN architecture from scratch using PyTorch
- Train the model on a real-world benchmark dataset (CIFAR-10)
- Evaluate model performance globally and per class
- Save and reload the trained model
- Perform inference on external images
- Compare training performance across different hardware devices

---

## 🛠️ Technology Stack
- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
- Deep Learning (CNNs)  

---

## ⚙️ Pipeline Overview

### 1️⃣ Environment & Hardware Selection
The script automatically detects and selects the most powerful available hardware:
- NVIDIA GPU (CUDA)
- Apple GPU (MPS)
- CPU (fallback)

This ensures optimal training performance across different systems.

---

### 2️⃣ Data Loading & Preprocessing
- Training and test datasets are loaded from CIFAR-10
- Images are converted to tensors
- Pixel values are normalized to the range **[-1.0, 1.0]**, improving training stability
- Data is loaded in batches (`batch_size = 64`)

---

### 3️⃣ Model Architecture — ConvNet
The CNN architecture consists of:

- **2 Convolutional layers**  
  → Feature extraction (edges, textures, shapes)  
- **Max-Pooling layers**  
  → Dimensionality reduction  
- **3 Fully Connected (Linear) layers**  
  → Final classification decision  

This structure mimics how visual patterns are progressively learned in deep learning models.

---

### 4️⃣ Training Loop
- Training runs for **10 epochs**
- Forward pass: predictions are generated
- Loss function: **CrossEntropyLoss**
- Optimization: **Adam optimizer**
- Backward pass updates model weights
- Accuracy is evaluated on the test set after each epoch

---

### 5️⃣ Model Evaluation
After training, the model is evaluated on the test dataset:
- Overall accuracy
- Accuracy per class (all 10 CIFAR-10 categories)
- Performance analysis and interpretation

---

### 6️⃣ Model Saving & Deployment
- The trained model is saved to disk (`.pth`)
- The model is reloaded for inference
- External images (outside the dataset) are classified
- The predicted class and confidence score are displayed
- Tests include images from **unseen classes**

---

### 7️⃣ Performance Comparison
The project concludes with a comparison of execution time across:
- GPU
- CPU
- Different hardware environments

---

## 📁 Project Structure
```

├── model.pth
├── main.py
├── utils.py
└── README.md

```

---

## 📈 Key Results
- Successful training of a CNN on CIFAR-10
- Accurate image classification across multiple classes
- Clear understanding of CNN behavior and limitations
- Demonstration of real-world inference and deployment

---

## 💡 Business & Technical Value
This project demonstrates:
- End-to-end Deep Learning workflow
- Practical understanding of CNNs
- Ability to deploy and evaluate AI models
- Awareness of hardware acceleration and performance

It is an excellent foundation for roles in:
- **Machine Learning**
- **Artificial Intelligence**
- **Computer Vision**
- **Data Science**

---

## 📌 Disclaimer
This project is **educational** and uses a public dataset to demonstrate deep learning techniques.

---

## 🇧🇷 Versão em Português

## 📌 Visão Geral do Projeto
Este projeto implementa um **pipeline completo de Deep Learning** para **classificação de imagens** utilizando **PyTorch**.

O objetivo central é construir, treinar, avaliar e utilizar uma **Rede Neural Convolucional (CNN)** — chamada **ConvNet** — para classificar imagens do **dataset CIFAR-10**, composto por **10 categorias** como avião, carro, pássaro, gato, cachorro, entre outras.

📂 **Fonte dos Dados:**  
https://www.cs.toronto.edu/~kriz/cifar.html

---

## 🧠 Contexto do Problema
A classificação de imagens é um dos pilares da **Visão Computacional** e uma aplicação fundamental da **Inteligência Artificial**.

Este projeto demonstra, de forma prática, como modelos de Deep Learning aprendem padrões visuais diretamente a partir dos pixels das imagens.

---

## 🎯 Objetivos do Projeto
- Construir uma CNN do zero com PyTorch
- Treinar o modelo em um dataset real (CIFAR-10)
- Avaliar desempenho geral e por classe
- Salvar e reutilizar o modelo treinado
- Realizar inferência em imagens externas
- Comparar desempenho entre diferentes hardwares

---

## 🛠️ Tecnologias Utilizadas
- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
- Deep Learning (CNNs)  

---

## ⚙️ Visão Geral do Pipeline

### 1️⃣ Seleção de Hardware
O script identifica automaticamente o melhor hardware disponível:
- GPU NVIDIA
- GPU Apple
- CPU

---

### 2️⃣ Carregamento e Pré-processamento dos Dados
- Conversão das imagens em tensores
- Normalização dos pixels para **[-1.0, 1.0]**
- Organização dos dados em batches (`batch_size = 64`)

---

### 3️⃣ Arquitetura do Modelo — ConvNet
- 2 camadas convolucionais para extração de padrões visuais  
- Camadas de max-pooling para redução dimensional  
- 3 camadas totalmente conectadas para classificação final  

---

### 4️⃣ Treinamento
- Treinamento por **10 épocas**
- Função de perda: **CrossEntropyLoss**
- Otimizador: **Adam**
- Avaliação da acurácia ao final de cada época

---

### 5️⃣ Avaliação do Modelo
- Acurácia geral
- Acurácia por classe
- Análise detalhada dos resultados

---

### 6️⃣ Salvamento e Uso do Modelo
- Modelo salvo em `.pth`
- Inferência em imagens externas
- Exibição da previsão e nível de confiança
- Testes com classes não vistas no treinamento

---

### 7️⃣ Comparação de Performance
- Comparação de tempo de execução entre CPU e GPU

---

## 📈 Resultados e Aprendizados
- Treinamento completo de uma CNN
- Compreensão prática de Deep Learning
- Aplicação real de IA
- Noções de deploy e performance

---

## 📌 Observação
Projeto com fins **educacionais**, utilizando dados públicos para demonstrar conceitos reais de Inteligência Artificial.
