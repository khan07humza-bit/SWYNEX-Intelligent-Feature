import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("customer_support_tickets.csv")

X = data["customer_message"]
y = data["category"]

# Build and train model
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

CONFIDENCE_THRESHOLD = 0.35

# Evaluation examples
test_cases = [
    ("My package has still not arrived", "Delivery Issue"),
    ("I was charged twice for my purchase", "Payment Issue"),
    ("The item I received is broken", "Product Issue"),
    ("I want my money back", "Refund / Return"),
    ("I forgot my account password", "Other"),
]

print("=" * 65)
print("MODEL EVALUATION EXAMPLES")
print("=" * 65)

correct = 0

for message, expected in test_cases:

    probabilities = model.predict_proba([message])[0]
    best_index = probabilities.argmax()

    prediction = model.classes_[best_index]
    confidence = probabilities[best_index]

    if confidence < CONFIDENCE_THRESHOLD:
        final_result = "Needs Human Review"
    else:
        final_result = prediction

    is_correct = prediction == expected

    if is_correct:
        correct += 1

    print(f"\nInput: {message}")
    print(f"Expected: {expected}")
    print(f"Model Prediction: {prediction}")
    print(f"Final Result: {final_result}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Evaluation: {'PASS' if is_correct else 'FAIL'}")


print("\n" + "=" * 65)
print(f"Correct Predictions: {correct}/{len(test_cases)}")
print("=" * 65)


# Failure / uncertainty examples
failure_cases = [
    "I have a problem",
    "Something is wrong",
    "Please help me"
]

print("\nFAILURE / UNCERTAINTY EXAMPLES")
print("=" * 65)

for message in failure_cases:

    probabilities = model.predict_proba([message])[0]
    best_index = probabilities.argmax()

    prediction = model.classes_[best_index]
    confidence = probabilities[best_index]

    print(f"\nInput: {message}")
    print(f"Suggested Category: {prediction}")
    print(f"Confidence: {confidence * 100:.2f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        print("Final Result: Needs Human Review")
    else:
        print(f"Final Result: {prediction}")
