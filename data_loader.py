import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("sinatavakoli/books-dataset-for-nlp-and-recommendation-systems")
data_file = Path(path) / "Books.csv"  # Adjust filename as needed
