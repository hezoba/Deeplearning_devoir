#Importation des biblothèque necesaire
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import gradio as gr
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/My Drive/Classification')
from Classification_langue import BERTSequenceClassifier

# Chargement des fichiers
data_path = '/content/drive/My Drive/Classification/bert_sequence_classifier.pth'
data = pd.read_csv('/content/drive/My Drive/Classification/contradictory-my-dear.csv')

#Actualisation des labels

dico_mapping = {}

categories_uniques = data['language'].unique()
for i, categorie in enumerate(categories_uniques):
    dico_mapping[categorie] = i 

data['label'] = data['language'].map(dico_mapping)

# Vérifier si le GPU est disponible, sinon utiliser le CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chargement du tokenizer et du modèle entraîné
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERTSequenceClassifier(num_labels=len(data['label'].unique())).to(device)
model.load_state_dict(torch.load(data_path))
model.eval()

# Définition de la fonction de prédiction de la langue
def predict_language(text):
    if text.lower() == "effacer":
        return ""

    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt').to(device)
    attention_mask = input_ids != 0
    outputs = model(input_ids, attention_mask)
    predicted_label = torch.argmax(outputs[0]).item()

    return data['language'].unique()[predicted_label]

# Création de l'interface Gradio pour l'API
Interface = gr.Interface(fn=predict_language, inputs="text", outputs="text", title="API de prédiction de langue", description="Entrez le texte à prédire")

# Lancement de l'API
Interface.launch(share=True)