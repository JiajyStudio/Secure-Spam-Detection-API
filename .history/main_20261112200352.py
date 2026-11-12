import kagglehub
import pandas as pd
from pathlib import Path

path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("Path to dataset files:", path)