# Import des bibliothèques
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Charger le fichier CSV en supprimant les incohérences

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/My Drive/Classification/contradictory-my-dear.csv')

#Actualisation des labels

dico_mapping = {}

categories_uniques = data['language'].unique()
for i, categorie in enumerate(categories_uniques):
    dico_mapping[categorie] = i

data['label'] = data['language'].map(dico_mapping)

# Concaténation de 'premise' et 'hypothesis' pour former le texte
data['text'] = data['premise'] + " " + data['hypothesis']

# Définition du jeu de données personnalisé
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.loc[index, 'text'])
        label = self.data.loc[index, 'label']

        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initialisation du tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128
dataset = CustomDataset(data, tokenizer, max_length)

# Division train-test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Définition du modèle de classification de séquences BERT
class BERTSequenceClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(BERTSequenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Initialisation du modèle
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BERTSequenceClassifier(num_labels=len(data['label'].unique())).to(device)

# Entraînement du modèle
num_epochs = 5  # Nombre d'époques
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
criterion = torch.nn.CrossEntropyLoss()

model.train()
total_iterations = 0

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids, labels=labels)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        total_iterations += 1
        if total_iterations % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Total Iterations: {total_iterations}, Loss: {total_loss/100:.4f}')
            total_loss = 0.0

# Sauvegarde du modèle entraîné

torch.save(model.state_dict(), 'bert_sequence_classifier.pth')
print("Modèle sauvegardé avec succès.")

# Évaluation du modèle
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy}')