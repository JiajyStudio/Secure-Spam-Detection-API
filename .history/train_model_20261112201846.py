import kagglehub
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("Path to dataset files:", path)

# Load the CSV file
csv_file = Path(path) / "spam.csv"

# Load the dataset
df = pd.read_csv(csv_file, encoding='latin-1')

print("Dataset head:", df.head())
print("Dataset shape:", df.shape)

# ===================
# Data Preprocessing
# ===================

# Drop unnecessary columns
df = df[['v1', 'v2']]

# Rename columns for clarity
df = df.rename(columns={
    'v1': 'label',
    'v2': 'message'
})

# check missing values
print("Missing values in dataset:", df.isnull().sum())

# remove missing values
df = df.dropna(subset=["label", "message"])

# ===================
# Split the Dataset (80/20)
# ===================

# Feature (X) and target (y)
x = df['message']
y = df['label']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)

print("Training sample:", len(x_train))
print("Testing sample:", len(x_test))

# ==================
# ML Pipeline
# ==================

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(x_train, y_train)

print("Model training completed.")