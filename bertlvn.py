import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score

# Define Dataset Class
class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # Ensure text is a string
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'  # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        }


# Define BERT Classification Model
class LegalBertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels, dropout_rate=0.3):
        super(LegalBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, num_labels)


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use max pooling on the last hidden state
        pooled_output = torch.max(outputs.last_hidden_state, dim=1)[0]

        x = self.dropout(pooled_output)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout again after the first linear layer
        x = self.linear2(x)
        return x


# --- Main Execution ---

# 1. Load Data
csv_file = "C:\\Users\\collinguo\\bertlegalvsnot\\legal_text_classification.csv"  # Correct path here
df = pd.read_csv(csv_file)


# Print first few rows to check the data loaded correctly
print("First 5 rows of the dataframe:")
print(df.head())


# 2. Preprocess Data
TEXT_COLUMN = 'case_text'  # Or the name of your text column
LABEL_COLUMN = 'case_outcome' # Name of the outcome column


# Check for null values
print("\nNumber of null values in each column:")
print(df.isnull().sum())

#Handle missing values.  This is a simple approach.  More sophisticated methods might be better depending on your data.
df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]) # remove rows with missing text or label

texts = df[TEXT_COLUMN].tolist()
labels = df[LABEL_COLUMN].tolist()


# Convert labels to numerical values.  This is crucial for classification.
label_map = {label: i for i, label in enumerate(sorted(set(labels)))}  # Create mapping
numerical_labels = [label_map[label] for label in labels]
num_labels = len(label_map) # number of distinct categories/labels
print(f"\nLabel Mapping: {label_map}")


# 3. Prepare Data for BERT
BERT_MODEL_NAME = 'bert-base-uncased'  # Or any other BERT model
MAX_LENGTH = 128  # Adjust as needed.  Try different lengths.

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, numerical_labels, test_size=0.2, random_state=42
)


# Create Datasets
train_dataset = LegalDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = LegalDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)


# Create DataLoaders
BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# 4. Model Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LegalBertClassifier(BERT_MODEL_NAME, num_labels).to(device)

# Optimizer and Loss Function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Training Loop
EPOCHS = 30  # Adjust as needed. Start with a small number and increase.
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_dataloader)}")


# 5. Model Evaluation
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())


# 6. Performance Metrics
print("Classification Report:")
print(classification_report(all_labels, all_predictions, target_names=label_map.keys()))
print("Accuracy:", accuracy_score(all_labels, all_predictions))