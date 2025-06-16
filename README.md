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

