import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset safely
try:
    data = pd.read_csv("customer_support_tickets.csv")
except FileNotFoundError:
    print("Error: customer_support_tickets.csv was not found.")
    exit()

# Validate dataset
required_columns = {"customer_message", "category"}

if not required_columns.issubset(data.columns):
    print("Error: Dataset must contain customer_message and category columns.")
    exit()

# Prepare data
X = data["customer_message"]
y = data["category"]

# Build ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

# Train model
model.fit(X, y)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.35


def classify_message(message):
    """
    Classify a customer message and apply confidence-based
    human review when the prediction is uncertain.
    """

    # Error handling for empty input
    if message is None or not message.strip():
        return {
            "status": "error",
            "message": "Please enter a valid customer message."
        }

    # Prediction
    probabilities = model.predict_proba([message])[0]
    best_index = probabilities.argmax()

    predicted_category = model.classes_[best_index]
    confidence = probabilities[best_index]

    # Intelligent feature:
    # Send uncertain predictions for human review
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "status": "human_review",
            "category": "Needs Human Review",
            "suggested_category": predicted_category,
            "confidence": confidence
        }

    return {
        "status": "success",
        "category": predicted_category,
        "confidence": confidence
    }


print("=" * 60)
print("   INTELLIGENT CUSTOMER SUPPORT TICKET CLASSIFIER")
print("=" * 60)

print("\nModel trained successfully.")
print("Low-confidence predictions are sent for human review.")
print("Type 'exit' to close the program.")

while True:

    message = input("\nCustomer Message: ")

    if message.lower().strip() == "exit":
        print("Classifier closed.")
        break

    result = classify_message(message)

    if result["status"] == "error":

        print("Error:", result["message"])

    elif result["status"] == "human_review":

        print("Result: Needs Human Review")
        print(
            "Suggested Category:",
            result["suggested_category"]
        )
        print(
            f"Confidence: {result['confidence'] * 100:.2f}%"
        )
        print(
            "Reason: Model confidence is below the "
            "35% threshold."
        )

    else:

        print(
            "Predicted Category:",
            result["category"]
        )
        print(
            f"Confidence: {result['confidence'] * 100:.2f}%"
        )
