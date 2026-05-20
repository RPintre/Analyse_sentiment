import os
import re
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

# DOC BERT : https://huggingface.co/docs/transformers/v4.48.2/en/model_doc/bert#transformers.BertForSequenceClassification
# DOC AutoTokenizer : https://huggingface.co/docs/transformers/v4.48.2/en/model_doc/bert#transformers.BertForSequenceClassification
# DOC pandas : https://pandas.pydata.org/docs/user_guide/
# DOC torch : https://docs.pytorch.org/docs/2.12/torch.html
# DOC sklearn Train_test_split : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html



# Dataset path
base_data_dir = os.path.join(os.getcwd(), 'kaggle', 'input')
data_file = os.path.join(base_data_dir, 'tweets-dataset', 'tweets.csv')
if not os.path.exists(data_file):
    raise FileNotFoundError(
        f"Dataset not found at {data_file}. Place tweets.csv at `{data_file}` or update the path."
    )

# Load dataset
data = pd.read_csv(data_file, encoding='latin')[['Target', 'Text']].dropna()

# Convert sentiment labels: 0 = negative, 4 -> 1 = positive
data['Target'] = data['Target'].replace(4, 1)

data = data[data['Target'].isin([0, 1])].reset_index(drop=True)

# Take 50 samples of each class (100 total, 50% negative, 50% positive)
samples_per_class = 1000
balanced_data = pd.concat(
    [data[data['Target'] == label].sample(samples_per_class, random_state=42)
     for label in sorted(data['Target'].unique())]
)

data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

print('Class counts ('+ str(samples_per_class*2) +' total, 50/50):')
print(data['Target'].value_counts())

# Minimal text cleaning for BERT

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

texts = data['Text'].astype(str).apply(clean_text).tolist()
labels = data['Target'].astype(int).tolist()

# Train/test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

# Tokenize with AutoTokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 128

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='pt',
)
val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='pt',
)

train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(train_labels, dtype=torch.long),
)
val_dataset = TensorDataset(
    val_encodings['input_ids'],
    val_encodings['attention_mask'],
    torch.tensor(val_labels, dtype=torch.long),
)

batch_size = 16
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print('Using device:', device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# Training loop

def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc='Training', leave=False)
    for batch in progress:
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]
        model.zero_grad()
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def eval_model(model, dataloader):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0.0

    progress = tqdm(dataloader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for batch in progress:
            input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, axis=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, predictions, true_labels

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_loss, preds, true_labels = eval_model(model, val_loader)
    accuracy = accuracy_score(true_labels, preds)
    print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={accuracy:.4f}")

print('\nValidation results:')
print(classification_report(true_labels, preds, target_names=['negative', 'positive']))
print('Confusion matrix:')
print(confusion_matrix(true_labels, preds))

# Example prediction
sample_text = ['this article is very beautiful']
sample_encoding = tokenizer(
    sample_text,
    truncation=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='pt',
).to(device)
model.eval()
with torch.no_grad():
    outputs = model(**sample_encoding)
    prediction = torch.argmax(outputs.logits, axis=1).item()

print("Sample prediction:", 'Positive' if prediction == 1 else 'Negative')
