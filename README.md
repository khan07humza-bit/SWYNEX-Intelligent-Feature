# SWYNEX Intelligent Feature – Task 3

## Intelligent Customer Support Ticket Classifier

This project is Task 3 of my internship with SWYNEX Technologies. It improves the Customer Support Ticket Classification prototype developed in Task 2 by adding confidence-based decision making, error handling, evaluation examples, and failure-case handling.

## Project Objective

The system automatically classifies customer support messages into five categories:

- Payment Issue
- Delivery Issue
- Product Issue
- Refund / Return
- Other

The main improvement in Task 3 is that the system does not blindly trust every prediction.

## Intelligent Feature

The classifier calculates a confidence score for every prediction.

If the confidence is **35% or higher**, the predicted category is returned.

If the confidence is **below 35%**, the ticket is marked as:

`Needs Human Review`

The model still provides its suggested category and confidence score so that a support agent can review the result.

### Example

Input:

`I have a problem`

Possible Output:

`Needs Human Review`

This approach helps prevent uncertain AI predictions from being automatically treated as reliable decisions.

## Error Handling

The prototype includes basic error handling.

### Empty Input

If the user enters an empty message, the system returns:

`Please enter a valid customer message.`

### Missing Dataset

If `customer_support_tickets.csv` cannot be found, the program displays an error message and stops safely.

### Invalid Dataset

The program also checks whether the required `customer_message` and `category` columns exist before training the model.

## Model

The prototype uses:

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

Workflow:

`Customer Message → TF-IDF → Logistic Regression → Confidence Check → Prediction or Human Review`

## Dataset

The prototype uses a small labeled dataset containing 50 customer support messages.

There are 10 examples for each of the five supported categories.

## Evaluation Examples

The `evaluation.py` file contains predefined examples with known expected categories.

For each example, it displays:

- Input message
- Expected category
- Model prediction
- Confidence score
- Final result
- PASS or FAIL

This makes the model's behavior easier to evaluate instead of relying only on individual demonstrations.

## Failure Cases

The evaluation also includes intentionally vague messages such as:

- `I have a problem`
- `Something is wrong`
- `Please help me`

These examples demonstrate cases where the model may not have enough information to make a reliable classification.

Low-confidence cases can be routed to human review rather than automatically accepted.

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Intelligent Classifier

```bash
python intelligent_classifier.py
```

Enter a customer support message when prompted.

Type:

`exit`

to close the program.

## Run Evaluation Examples

```bash
python evaluation.py
```

The program will display the actual predictions, confidence scores, PASS/FAIL results, and uncertainty examples.

## Limitations

This is a small internship prototype and not a production-ready customer support system.

- The dataset contains only 50 manually created examples.
- The confidence threshold is a prototype rule and has not been calibrated on production data.
- Only five categories are supported.
- Ambiguous or unfamiliar messages may still be classified incorrectly.
- A larger labeled dataset and proper held-out evaluation would be needed before real-world deployment.

## Future Improvements

Future versions could include:

- A larger real-world dataset
- A web-based user interface
- Additional support categories
- Better confidence calibration
- Train/validation/test evaluation
- Automatic routing to customer support teams
- Human feedback for improving future predictions

---

## Internship

**Organization:** SWYNEX Technologies  
**Task:** Task 3 – Intelligent Feature  
**Domain:** Artificial Intelligence / Machine Learning
