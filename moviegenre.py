import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

train_path = r'C:\Users\naval\OneDrive\Desktop\movie_genre_classication\archive\Genre Classification Dataset\train_data.txt'
train_data = pd.read_csv(train_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')

test_path = r'C:\Users\naval\OneDrive\Desktop\movie_genre_classication\archive\Genre Classification Dataset\test_data.txt'
test_data = pd.read_csv(test_path, sep=':::', names=['Id', 'Title', 'Description'], engine='python')

print(train_data.describe())
print(test_data.describe())
print(train_data.isnull().sum())
print(test_data.isnull().sum())

class_distribution = train_data['Genre'].value_counts()
print('Class Distribution:')
print(class_distribution)
imbalance_ratio = class_distribution.min() / class_distribution.max()
print('Imbalance Ratio:', imbalance_ratio)
