
# Movie Genre Classification

This project aims to classify movie genres based on their descriptions using machine learning techniques. The dataset includes movie titles, genres, and descriptions for training and testing.

## Project Structure

- `train_data.txt`: Training dataset.
- `test_data.txt`: Test dataset.
- `movie_genre_classification.py`: Python script for preprocessing, training, and testing the model.
- `predicted_genres.csv`: Output file with predicted genres for the test dataset.
- `tfidf_vectorizer.pkl`: Pickled TF-IDF vectorizer model.
- `nb_classifier.pkl`: Pickled Naive Bayes classifier model.

## Setup and Installation

1. Clone the repository to your local machine.
2. Ensure you have the required Python libraries installed. You can install them using the following command:
   ```bash
   pip install pandas matplotlib seaborn nltk scikit-learn

## Data Preprocessing

1. **Load the Data:**
   - Load the training and test datasets from text files using `pd.read_csv`:
     ```python
     train_path = r"C:\Users\naval\OneDrive\Desktop\movie_genre_classification\archive\Genre Classification Dataset\train_data.txt"
     train_data = pd.read_csv(train_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')
     test_path = r"C:\Users\naval\OneDrive\Desktop\movie_genre_classification\archive\Genre Classification Dataset\test_data.txt"
     test_data = pd.read_csv(test_path, sep=':::', names=['Id', 'Title', 'Description'], engine='python')
     ```

2. **Data Description:**
   - Use `describe()` to get a summary of the datasets:
     ```python
     train_data.describe()
     test_data.describe()
     ```
   - Check for missing values using `isnull().sum()`:
     ```python
     train_data.isnull().sum()
     test_data.isnull().sum()
     ```

3. **Class Distribution:**
   - Calculate and print the class distribution of the training data:
     ```python
     class_distribution = train_data['Genre'].value_counts()
     print("Class Distribution:")
     print(class_distribution)
     imbalance_ratio = class_distribution.min() / class_distribution.max()
     print("Imbalance Ratio:", imbalance_ratio)
     ```
   - Plot the class distribution using a bar chart:
     ```python
     plt.figure(figsize=(8, 6))
     class_distribution.plot(kind='bar', color='skyblue')
     plt.title('Class Distribution')
     plt.xlabel('Class')
     plt.ylabel('Frequency')
     plt.xticks(rotation=65)
     plt.show()
     ```

## Feature Extraction

- Use `TfidfVectorizer` to convert movie descriptions into TF-IDF features with a maximum of 5000 features:
  ```python
  tfidf_vectorizer = TfidfVectorizer(max_features=5000)
  X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['Description'])
  y_train = train_data['Genre']
## Model Training

1. **Train the Model:**
   - Train a Multinomial Naive Bayes classifier using the TF-IDF features and genres from the training dataset:
     ```python
     nb_classifier = MultinomialNB()
     nb_classifier.fit(X_train_tfidf, y_train)
     ```

2. **Evaluate the Model:**
   - Predict genres on the training dataset:
     ```python
     y_train_pred = nb_classifier.predict(X_train_tfidf)
     ```
   - Print the accuracy and classification report for the training set:
     ```python
     print("Accuracy on training set:", accuracy_score(y_train, y_train_pred))
     print("Classification Report on training set:\n", classification_report(y_train, y_train_pred))
     ```

## Model Testing

1. **Preprocess the Test Data:**
   - Convert the movie descriptions in the test dataset into TF-IDF features using the trained TF-IDF vectorizer:
     ```python
     X_test = tfidf_vectorizer.fit_transform(test_data['Description'])
     ```

2. **Predict and Save Results:**
   - Use the trained Naive Bayes classifier to predict genres for the test dataset:
     ```python
     X_test_predictions = nb_classifier.predict(X_test)
     test_data['Predicted_Genre'] = X_test_predictions
     test_data.to_csv('predicted_genres.csv', index=False)
     ```
   - Save the predictions to `predicted_genres.csv`.

## Model Serialization

- Serialize and save the trained TF-IDF vectorizer and Naive Bayes classifier using `pickle`:
  ```python
  import pickle
  with open('tfidf_vectorizer.pkl', 'wb') as file:
      pickle.dump(tfidf_vectorizer, file)
  with open('nb_classifier.pkl', 'wb') as file:
      pickle.dump(nb_classifier, file)
  print("Models pickled successfully.")
## Usage

1. **Run the Script:**
   ```bash
   python movie_genre_classification.py
## Example Output

```plaintext
# Example output:
Accuracy on training set: 0.95
Classification Report on training set:
               precision    recall  f1-score   support

    Action       0.93      0.92      0.92      1000
    Comedy       0.97      0.95      0.96      1000
    Drama        0.96      0.97      0.96      1000

    accuracy                           0.95      3000
   macro avg       0.95      0.95      0.95      3000
weighted avg       0.95      0.95      0.95      3000

