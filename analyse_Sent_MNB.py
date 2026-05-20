import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Téléchargement des ressources nécessaires pour le nettoyage (à faire une fois)
nltk.download('stopwords')

# 1. Dataset path & Loading
base_data_dir = os.path.join(os.getcwd(), 'kaggle', 'input')
data_file = os.path.join(base_data_dir, 'tweets-dataset', 'tweets.csv')
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Dataset not found at {data_file}.")

data = pd.read_csv(data_file, encoding='latin')[['Target', 'Text']].dropna()
data['Target'] = data['Target'].replace(4, 1)
data = data[data['Target'].isin([0, 1])].reset_index(drop=True)


# Échantillonnage
# samples_per_class = 1000000
# balanced_data = pd.concat([
#     data[data['Target'] == label].sample(samples_per_class, random_state=42)
#     for label in sorted(data['Target'].unique())
# ])
# data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Nettoyage AVANCÉ (Adapté pour MNB : Stopwords + Stemming)
stop_words = set(stopwords.words('english')) # Remplace par 'french' si tes tweets sont en français
stemmer = SnowballStemmer('english')

def clean_text_for_mnb(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # Supprime les URL
    text = re.sub(r'@\w+', ' ', text)              # Supprime les mentions
    text = re.sub(r'[^a-z\s]', ' ', text)          # Supprime la ponctuation et chiffres [cite: 34]
    
    # Tokenisation simple par espace 
    words = text.split()
    
    # Suppression des Stopwords  + Stemming (Racinisation) 
    cleaned_words = [stemmer.stem(w) for w in words if w not in stop_words]
    
    return ' '.join(cleaned_words)

print("Nettoyage du texte...")
texts = data['Text'].astype(str).apply(clean_text_for_mnb).tolist()
labels = data['Target'].astype(int).tolist()

# 3. Train/test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 4. Représentation des mots : TF-IDF 
# Transforme le texte en une matrice de nombres compréhensible par MNB
vectorizer = TfidfVectorizer(max_features=5000) 
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

# 5. Initialisation et entraînement du modèle MNB
print("Entraînement du modèle Multinomial Naive Bayes...")
model = MultinomialNB()
model.fit(X_train, train_labels)

# 6. Évaluation
preds = model.predict(X_val)
accuracy = accuracy_score(val_labels, preds)

print(f"\nValidation Accuracy: {accuracy:.4f}")
print('\nValidation results:')
print(classification_report(val_labels, preds, target_names=['negative', 'positive']))
print('Confusion matrix:')
print(confusion_matrix(val_labels, preds))

# 7. Exemple de prédiction
sample_text = ['this article is very beautiful']
# Attention : il faut appliquer le même nettoyage et la même transformation !
sample_cleaned = [clean_text_for_mnb(sample_text[0])]
sample_vector = vectorizer.transform(sample_cleaned)

prediction = model.predict(sample_vector)[0]
print("\nSample prediction:", 'Positive' if prediction == 1 else 'Negative')