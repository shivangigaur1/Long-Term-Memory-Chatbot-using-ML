import csv
import json
import pickle
import re
from collections import Counter, deque
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Predefined responses
PREDEFINED_RESPONSES = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! How can I help?",
    "sir": "Yes, I'm here to assist you.",
    "mam": "How can I help you, ma'am?",
    "excuse me": "Yes, how can I assist?",
    "sorry": "No problem! How can I assist you?",
    "hey": "Hey! How can I help you today?",
}

COMMON_QUESTION_KEYWORDS = ["what", "where", "who", "when", "why", "how"]

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Parse and preprocess data
def parse_and_preprocess_data(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        raw_entries = raw_data.get("data", [])
        parsed_data = []

        for entry in raw_entries:
            story = entry.get("story", "")
            questions = entry.get("questions", [])
            answers = entry.get("answers", [])
            additional_answers = entry.get("additional_answers", {})

            for question, answer in zip(questions, answers):
                question_text = preprocess_text(question.get("input_text", ""))
                answer_text = preprocess_text(answer.get("input_text", ""))
                turn_id = question.get("turn_id", None)

                alternatives = [
                    preprocess_text(alt_answer.get("input_text", ""))
                    for alt_answers in additional_answers.values()
                    for alt_answer in alt_answers
                    if alt_answer.get("turn_id") == turn_id
                ]

                parsed_data.append({
                    "story": preprocess_text(story),
                    "question": question_text,
                    "answer": answer_text,
                    "turn_id": turn_id,
                    "alternatives": alternatives
                })

        with open(output_file, 'wb') as f:
            pickle.dump(parsed_data, f)

        print(f"Parsed and preprocessed data saved to {output_file}.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

# Train the model
def train_model(input_file, output_model_file):
    try:
        with open(input_file, "rb") as f:
            processed_data = pickle.load(f)

        df = pd.DataFrame(processed_data)
        X = df["question"]
        y = df["answer"]

        vectorizer = TfidfVectorizer(max_features=5000)
        X_vectorized = vectorizer.fit_transform(X)

        class_counts = Counter(y)
        min_samples_required = 6
        filtered_classes = [cls for cls, count in class_counts.items() if count >= min_samples_required]
        filtered_mask = y.isin(filtered_classes)

        X_filtered = X_vectorized[filtered_mask]
        y_filtered = y[filtered_mask]

        min_class_size = min(Counter(y_filtered).values())
        k_neighbors = min(min_samples_required - 1, min_class_size - 1)

        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        with open(output_model_file, "wb") as f:
            pickle.dump((vectorizer, classifier), f)

        print(f"Model training complete. Saved as '{output_model_file}'.")
    except Exception as e:
        print(f"Error during training: {e}")

# Conversation Tracker
class ConversationTracker:
    def __init__(self, size=40):
        self.conversation_history = deque(maxlen=size)

    def add_conversation(self, question, answer):
        self.conversation_history.append((question, answer))
        self.save_to_csv(question, answer)

    def save_to_csv(self, question, answer):
        with open("track_chat.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([len(self.conversation_history), question, answer])

    @staticmethod
    def read_from_csv():
        conversation = []
        try:
            with open("track_chat.csv", mode="r") as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    conversation.append(row)
        except FileNotFoundError:
            pass
        return conversation

# Handle common questions
def handle_common_question(user_input):
    conversation = ConversationTracker.read_from_csv()
    common_word = user_input.split()[0]

    for row in reversed(conversation):
        _, question, answer = row
        if common_word in question.split():
            return answer
    return "Please clarify more."

# Handle subset questions
def handle_subset_question(user_input):
    conversation = ConversationTracker.read_from_csv()
    for row in reversed(conversation):
        _, question, answer = row
        if preprocess_text(user_input) in preprocess_text(question):
            return answer
    return "Please specify more."

# Voting system for the best answer
def get_best_answer(question, processed_data):
    answers = []
    for entry in processed_data:
        if preprocess_text(entry["question"]) == preprocess_text(question):
            answers.append(entry["answer"])
            answers.extend(entry.get("alternatives", []))

    if not answers:
        return None

    votes = Counter(answers)
    best_answer, _ = votes.most_common(1)[0]
    return best_answer

# Main response logic
def get_response(user_input, tracker, processed_data):
    user_input = preprocess_text(user_input)

    # 1. Predefined responses
    if user_input in PREDEFINED_RESPONSES:
        return PREDEFINED_RESPONSES[user_input]

    # 2. Proper question handling
    matching_entries = [entry for entry in processed_data if preprocess_text(entry["question"]) == user_input]
    if matching_entries:
        best_answer = get_best_answer(user_input, processed_data)
        if best_answer:
            tracker.add_conversation(user_input, best_answer)
            return best_answer

    # 3. Subset question handling
    subset_answer = handle_subset_question(user_input)
    if subset_answer != "Please specify more.":
        return subset_answer

    # # 4. Common question handling
    # if user_input.split()[0] in COMMON_QUESTION_KEYWORDS:
    #     return handle_common_question(user_input)

    # 5. Exit commands
    if user_input.lower() in ["exit", "bebye"]:
        return "Thank you! Goodbye. Have a nice day!"

    # 6. Incorrect/unclear query
    return "I'm sorry, I couldn't find a match. Please clarify more."

# Main chatbot loop
if __name__ == "__main__":
    tracker = ConversationTracker()

    with open("track_chat.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Question Number", "Question", "Answer"])

    parse_and_preprocess_data("data.json", "processed_data.pkl")
    train_model("processed_data.pkl", "qa_model.pkl")

    with open("processed_data.pkl", "rb") as f:
        processed_data = pickle.load(f)

    print("Hello! I am NADIA here to help you. Type ---'exit'--- or ---'bebye'--- to end the chat.")
    print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    while True:
        user_input = input("@ YOU--: ").strip()
        if user_input.lower() in ["exit", "bebye"]:
            print("NADIA : Thank you! Goodbye. Have a nice day!")
            break

        response = get_response(user_input, tracker, processed_data)
        print(f"NADIA  : {response}")
        print("---------------------------------------------------------------------------------------")
