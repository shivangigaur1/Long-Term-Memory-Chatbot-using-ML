#preprocessing python file
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Ensure 'results' directory exists
def ensure_results_dir():
    results_dir = 'D:/IIITD/ML_COURSE/results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

# Function to load and flatten the JSON dataset
def load_and_flatten_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    records = []
    for item in data['data']:
        for question in item['questions']:
            records.append({
                'story': item['story'],
                'source': item['source'],
                'question': question['input_text'],
                'turn_id': question['turn_id']
            })
    
    df = pd.DataFrame(records)
    return df

# Function to clean the data (remove duplicates, handle missing values)
def clean_data(df):
    df.dropna(inplace=True)  # Drop rows with missing values
    df.drop_duplicates(inplace=True)  # Drop duplicate rows
    return df

# Function to create additional features
def create_features(df):
    df['story_word_count'] = df['story'].apply(lambda x: len(x.split()))
    df['question_word_count'] = df['question'].apply(lambda x: len(x.split()))
    df['question_type'] = df['question'].apply(lambda x: x.split()[0])
    return df

# Function to perform basic EDA (visualizations)
def perform_eda(df):
    ensure_results_dir()  # Ensure the results directory exists
    
    # Story Word Count Distribution
    df['story_word_count'].hist(bins=30)
    plt.title('Story Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig('D:/IIITD/ML_COURSE/results/story_word_count_distribution.png')
    plt.show()

    # Question Type Distribution
    df['question_type'].value_counts().plot(kind='bar')
    plt.title('Question Type Distribution')
    plt.savefig('D:/IIITD/ML_COURSE/results/question_type_distribution.png')
    plt.show()

    # Generate Word Cloud
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(df['story']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('D:/IIITD/ML_COURSE/results/story_wordcloud.png')
    plt.show()

# Function to vectorize text data
def vectorize_text(df):
    X = df['story']
    y = df['question_type']
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate Naive Bayes model
def train_naive_bayes(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, y_pred))

# Function to train and evaluate Random Forest model
def train_random_forest(X_train, X_test, y_train, y_test):
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    
    y_pred_rf = clf_rf.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))

# Function to train and evaluate SVM model
def train_svm(X_train, X_test, y_train, y_test):
    clf_svm = SVC()
    clf_svm.fit(X_train, y_train)
    
    y_pred_svm = clf_svm.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))

# Main function to execute all steps
def main(file_path):
    # Step 1: Load and flatten data
    df = load_and_flatten_data(file_path)
    
    # Step 2: Clean the data
    df = clean_data(df)
    
    # Step 3: Create additional features
    df = create_features(df)
    
    # Step 4: Perform EDA
    perform_eda(df)
    
    # Step 5: Vectorize text data
    X_train, X_test, y_train, y_test = vectorize_text(df)
    
    # Step 6: Train and evaluate models
    train_naive_bayes(X_train, X_test, y_train, y_test)
    train_random_forest(X_train, X_test, y_train, y_test)  # Corrected: Closed parenthesis here
    train_svm(X_train, X_test, y_train, y_test)

# Run the main function with the path to your data.json file
if __name__ == '__main__':
    main('D:/IIITD/ML_COURSE/Chatbot/data.json')
