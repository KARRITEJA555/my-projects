import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB  # Use GaussianNB instead of MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def read_datasets():
    """ Reads users profile from csv files """
    genuine_users = pd.read_csv(r"C:\Users\KARRI TEJA\Desktop\project\fake-users.csv")
    fake_users = pd.read_csv(r"C:\Users\KARRI TEJA\Desktop\project\Genuine-users.csv")

    # Concatenate genuine and fake users data
    combined_data = pd.concat([genuine_users, fake_users])

    # Create labels for classification (1 for genuine users, 0 for fake users)
    y = [1] * len(genuine_users) + [0] * len(fake_users)

    return combined_data, y

# Load datasets using the read_datasets() function
data, y = read_datasets()

# Preprocess data if needed
data['description'].fillna('', inplace=True)
data['name'].fillna('', inplace=True)
data['followers_count'].fillna('', inplace=True)
data['friends_count'].fillna('', inplace=True)
data['location'].fillna('', inplace=True)

# Combine features
data['combined_features'] = data['name'] + ' ' + data['followers_count'].astype(str) + ' ' + data['friends_count'].astype(str) + ' ' + data['location'] + ' ' + data['description']

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and punctuations, and apply lemmatization
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha() and word.lower() not in stop_words]
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Preprocess the combined features
data['preprocessed_features'] = data['combined_features'].apply(preprocess_text)

# Split dataset into features and labels
X = data['preprocessed_features']
y = y

# Text vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)  # You can adjust the number of components as needed
X_pca = pca.fit_transform(X_vectorized.toarray())

# Split the PCA-transformed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train classifiers
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Use Gaussian Naive Bayes instead of Multinomial Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Calculate accuracies
svm_accuracy = accuracy_score(y_test, svm_classifier.predict(X_test))
nb_accuracy = accuracy_score(y_test, nb_classifier.predict(X_test))
knn_accuracy = accuracy_score(y_test, knn_classifier.predict(X_test))

# Save accuracies
accuracies = {'SVM': svm_accuracy, 'Naive Bayes': nb_accuracy, 'KNN': knn_accuracy}

# Get the classifier with the highest accuracy
best_classifier = max(accuracies, key=accuracies.get)

# Print accuracies of all classifiers
st.write("Accuracies:")
for clf_name, accuracy in accuracies.items():
    st.write(f"{clf_name} Classifier Accuracy: {accuracy}")

# Plotting bar chart for accuracies
fig, ax = plt.subplots()
ax.bar(accuracies.keys(), accuracies.values())
ax.set_xlabel('Classifier')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison of Classifiers')
st.pyplot(fig)

# Save vectorizer and classifiers
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(svm_classifier, 'svm_classifier.pkl')
joblib.dump(nb_classifier, 'nb_classifier.pkl')
joblib.dump(knn_classifier, 'knn_classifier.pkl')

# Define the Streamlit app
def main():
    st.title('Fake Profile Identification')

    # Text input for the user to enter profile description
    name = st.text_input('Enter name:')
    followers_count = st.text_input('Enter followers count:')
    friends_count = st.text_input('Enter friends count:')
    location = st.text_input('Enter location:')
    description = st.text_area('Enter profile description:')

    # Prediction button
    if st.button('Predict'):
        # Load vectorizer and classifiers
        vectorizer = joblib.load('vectorizer.pkl')
        if best_classifier == 'SVM':
            classifier = joblib.load('svm_classifier.pkl')
        elif best_classifier == 'Naive Bayes':
            classifier = joblib.load('nb_classifier.pkl')
        else:
            classifier = joblib.load('knn_classifier.pkl')

        # Preprocess the input description
        combined_features = name + ' ' + followers_count + ' ' + friends_count + ' ' + location + ' ' + description
        description_vectorized = vectorizer.transform([preprocess_text(combined_features)])
        
        # Apply PCA to the input data for prediction
        description_pca = pca.transform(description_vectorized.toarray())

        # Predict using the best classifier
        prediction = classifier.predict(description_pca)
        st.write("Prediction:", "Genuine Profile" if prediction[0] == 1 else "Fake Profile")

        # Print confusion matrix of all classifiers
        st.subheader("Confusion Matrices:")
        for clf_name, clf in [('SVM', svm_classifier), ('Naive Bayes', nb_classifier), ('KNN', knn_classifier)]:
            st.subheader(f"{clf_name} Classifier:")
            st.write(confusion_matrix(y_test, clf.predict(X_test)))


        # Print classification report of the best classifier
        st.subheader(f"Classification Report ({best_classifier} Classifier):")
        st.write(classification_report(y_test, classifier.predict(X_test)))


if _name_ == '_main_':
    main()