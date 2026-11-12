import kagglehub
import pandas as pd
from pathlib import Path

path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("Path to dataset files:", path)

# Load the CSV file
csv_file = Path(path) / "spam.csv"

# Load the dataset
df = pd.read_csv(csv_file, encoding='latin-1')

print("Dataset head:", df.head())
print("Dataset shape:", df.shape)