PS D:\IIITD\ML_COURSE\project\project18> python chatbot.py
Parsed and preprocessed data saved to processed_data.pkl.
Classification Report:
              precision    recall  f1-score   support

          15       0.99      1.00      1.00       198
       china       1.00      0.99      1.00       188
         cnn       1.00      1.00      1.00       191
        dick       1.00      1.00      1.00       165
       eight       1.00      1.00      1.00       189
     england       1.00      1.00      1.00       172
       false       1.00      1.00      1.00       172
        five       1.00      1.00      1.00       187
     florida       1.00      1.00      1.00       172
        four       0.97      0.85      0.91       173
      friday       1.00      1.00      1.00       179
       harry       1.00      1.00      1.00       188
          no       0.69      0.58      0.63       153
         one       1.00      1.00      1.00       182
       peter       1.00      1.00      1.00       188
       seven       1.00      1.00      1.00       188
         six       1.00      0.99      1.00       171
    the park       0.99      1.00      1.00       182
       three       0.99      0.91      0.95       172
         tom       1.00      0.99      1.00       161
         two       0.80      0.99      0.88       162
     unknown       0.99      0.99      0.99       147
   wednesday       1.00      1.00      1.00       171
         yes       0.61      0.72      0.66       145

    accuracy                           0.96      4196
   macro avg       0.96      0.96      0.96      4196
weighted avg       0.97      0.96      0.96      4196

1. Classes
The leftmost column lists the classes (e.g., 15, china, cnn, yes, etc.) that your model predicts.
Each of these represents a possible label or answer your model is trained to predict for a given question.


2. Metrics

The columns precision, recall, f1-score, and support evaluate your model's performance for each class.

a. Precision
Precision = True Positives / (True Positives + False Positives)
It measures how many of the answers your model predicted for a class were correct.
Example: For china, a precision of 1.00 means that every time the model predicted china, it was correct.
b. Recall
Recall = True Positives / (True Positives + False Negatives)
It measures how many of the actual answers in a class your model was able to predict.
Example: For china, a recall of 0.99 means that the model identified 99% of the actual china answers correctly.
c. F1-Score
F1-Score is the harmonic mean of precision and recall.
F1 = 2 × (Precision × Recall) / (Precision + Recall)
A higher F1-score indicates a better balance between precision and recall.
Example: china has an F1-score of 1.00, meaning both precision and recall are excellent.
d. Support
Support is the number of occurrences of each class in the test data.
Example: china appears 188 times in the test dataset.


3. Overall Metrics

At the bottom of the report, you see overall metrics for the model:

a. Accuracy
Accuracy = Total Correct Predictions / Total Predictions
Accuracy = 0.96 means the model correctly predicted 96% of the total test data.
b. Macro Average
Macro average computes the average of precision, recall, and F1-score across all classes without considering class imbalance.
Good for evaluating the model's performance across all classes equally.
c. Weighted Average
Weighted average computes the average of precision, recall, and F1-score, weighted by the support of each class.
It considers the class imbalance by giving more weight to classes with more samples.


4. Observations

High-Performing Classes: Classes like china, friday, seven, yes, unknown, etc., have high precision, recall, and F1-scores, indicating excellent performance.
Low-Performing Classes: Classes like yes (F1 = 0.66) and no (F1 = 0.63) show room for improvement, likely due to their lower precision or recall.
This could be because these classes have more overlap with other answers, making them harder to predict.
Class Imbalance: Some classes like two or no might suffer from low support or ambiguity in questions, which impacts performance.


5. Why This Matters

The report helps identify which classes are easier or harder for your model to predict.
This insight can guide further improvement:
Consider augmenting data for low-performing classes (yes, no).
Focus on misclassified examples to improve model training.

Description of Your Long-Term Memory Chat Model
Your chatbot model is designed to simulate long-term memory in conversations. By integrating natural language processing (NLP), machine learning, and memory-tracking mechanisms, the chatbot can retain, reference, and utilize prior conversations and dataset knowledge to deliver context-aware and dynamic responses. Here's a detailed description of how your model achieves long-term memory:

Core Components of the Model
Memory Architecture

Conversation Tracker:
Maintains a conversation history using a deque structure, which is efficient for managing and querying recent exchanges.
Stores all user inputs and chatbot responses in a CSV file (track_chat.csv), creating a persistent log of interactions.
Allows the chatbot to recall prior questions and answers, enabling follow-up queries and context-aware responses.
Dynamic Context Matching:
The chatbot can match a user's new query with previously asked questions or topics to provide consistent and relevant responses.
Machine Learning Models

TfidfVectorizer:
Converts questions into numerical vectors, capturing the significance of words relative to the entire dataset.
Enables similarity matching between current and prior questions.
RandomForestClassifier:
Serves as the main model for question classification and answer prediction.
Predicts the most appropriate answer based on patterns learned from training data.
SMOTE:
Balances the training dataset, ensuring that the model does not overfit on frequently occurring answers.
Predefined Responses

Handles standard phrases like greetings or apologies.
Examples: "hi", "hello", "sorry", etc.
Provides instant, predictable responses to common inputs.
Dataset Integration

The chatbot is trained on a dataset containing stories, questions, and answers. It can use this knowledge base to answer direct or related questions.
Multiple layers of answers (exact matches, subset questions, alternative interpretations) make the responses robust and adaptable.
Persistent Memory (File-Based)

Uses .pkl files to store:
Processed data (processed_data.pkl): Parsed and cleaned dataset for quick access.
Trained models (qa_model.pkl): Trained machine learning models for prediction without retraining.
Voting System

When multiple possible answers exist for a question (from dataset or alternatives), the model uses a voting mechanism to determine the most common and likely response.
Long-Term Memory Features
Persistent Context Awareness

Stores all past user queries and chatbot responses in track_chat.csv.
Queries are matched against the history to maintain conversational flow:
For example, if the user asks, "What is the capital of France?" and later says, "What about Germany?" the bot references the prior question to infer context.
Follow-Up and Subset Queries

Handles follow-up questions effectively:
If a user first asks, "What color is the sky?" and later says, "And during sunsets?", the bot uses the stored conversation to provide a relevant response.
Recognizes subsets of previous questions, enabling answers to partially repeated or related queries.
Dynamic Learning

The chatbot continuously updates the conversation history during runtime, mimicking human-like memory.
Uses historical data to refine its understanding of ambiguous or repeated questions.
Workflow: How the Chatbot Processes Inputs
Preprocessing User Input

Converts the user’s input to lowercase and removes punctuation to standardize text.
Identifies whether the input is:
A predefined phrase.
A direct match for a dataset question.
A follow-up or related query.
Memory Integration

Searches the conversation history for matching or related questions.
Uses the ConversationTracker to retrieve stored responses or context.
Question Matching and Prediction

If the user’s input matches a dataset question:
The model uses the RandomForestClassifier to predict the most likely answer.
Applies the voting system if multiple alternative answers exist.
If it’s a subset query:
Finds the closest matching previous question and provides its answer.
Response Generation

Generates responses using one of the following:
Predefined responses for common phrases.
Predicted answers from the trained model.
Answers retrieved from the conversation history.
If no match is found, the bot asks for clarification or encourages rephrasing.
Advantages of Long-Term Memory in This Chatbot
Enhanced Context Understanding

Tracks the entire conversation flow, ensuring responses are consistent with past exchanges.
Dynamic and Adaptable

Handles new queries by referencing historical data, improving its ability to respond to related or incomplete questions.
Efficient and Scalable

Uses .pkl files to store processed data and models, enabling fast initialization and efficient resource use.
User Engagement

Mimics human memory, making interactions feel natural and personalized.
Example Use Cases
Conversational Follow-Ups:

User: "Tell me about the story of Cotton."
Bot: "Cotton was a little white kitten who wanted to look like her orange sisters."
User: "Why did she want to look like them?"
Bot: "Because she felt sad about being different from her family."
Knowledge-Based QA:

User: "What is the capital of France?"
Bot: "The capital of France is Paris."
User: "What about Germany?"
Bot: "The capital of Germany is Berlin."
Subset Handling:

User: "Tell me the story of the grandmother bringing soup."
Bot: "The story is about a Chinese grandmother who brings soup to her neighbor recovering from surgery."
User: "What did she bring?"
Bot: "She brought soup, rice, vegetables, and sometimes meat or shrimp."
Conclusion
This chatbot effectively combines machine learning, NLP, and conversation tracking to simulate long-term memory. It ensures context-aware responses, handles follow-up questions dynamically, and leverages persistent storage to maintain continuity across sessions. This makes it ideal for applications in education, customer support, and interactive storytelling.