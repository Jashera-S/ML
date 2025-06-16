# ML


Sentiment using Naiye Bayes

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


texts = [
    "I love this product",         # Positive
    "This is amazing",             # Positive
    "I am very happy with it",     # Positive
    "I hate this",                 # Negative
    "This is terrible",            # Negative
    "I am disappointed",           # Negative
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = Positive, 0 = Negative


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))


test_sentences = ["I love it", "This is bad", "I am not happy", "What a wonderful product"]
X_new = vectorizer.transform(test_sentences)
predictions = model.predict(X_new)

for sentence, label in zip(test_sentences, predictions):
    print(f"'{sentence}' → {'Positive' if label == 1 else 'Negative'}")

#spam ham using NB

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    'text': [
        "Win a brand new car",
        "Congratulations! You've won a free ticket",
        "Call now to claim your prize",
        "Hi, are we meeting today?",
        "Don't forget the appointment",
        "Lunch at 2?",
        "Get cheap meds online",
        "Exclusive deal just for you",
        "Are you coming to class?"
    ],
    'label': ['spam', 'spam', 'spam', 'ham', 'ham', 'ham', 'spam', 'spam', 'ham']
}

df = pd.DataFrame(data)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.3, random_state=1)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

new_messages = ["Congratulations, you won a trip!", "Hey, how are you?", "Free money now!", "Let’s go for dinner"]
new_vec = vectorizer.transform(new_messages)
preds = model.predict(new_vec)

for msg, pred in zip(new_messages, preds):
    print(f"'{msg}' → {'Spam' if pred == 1 else 'Ham'}")
    
