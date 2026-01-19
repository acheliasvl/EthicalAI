from data_cleaning import load_and_clean_data, split_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
import matplotlib.pyplot as plt

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.3f}")

def fairness_analysis(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if 'sex_Male' in X_test.columns:
        sensitive_feature = X_test['sex_Male']
    elif 'sex_Female' in X_test.columns:
        sensitive_feature = X_test['sex_Female']
    else:
        raise ValueError("Sex column not found!")

    metrics = {
        'accuracy': accuracy_score,
        'selection_rate': selection_rate,
        'true_positive_rate': true_positive_rate,
        'false_positive_rate': false_positive_rate,
    }

    metric_frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_feature)

    print("\nFairness Metrics by Gender:")
    print(metric_frame.by_group)

    metric_frame.by_group.plot.bar(subplots=True, layout=(2,2), legend=False, figsize=(10,8))
    plt.suptitle("Fairness Metrics by Gender")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_and_clean_data("adult_combined.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Training models...")
    models = train_models(X_train, y_train)

    print("\nModel Performance:")
    evaluate_models(models, X_test, y_test)

    # Fairness analysis with Random Forest model
    print("\nFairness Analysis for Random Forest Model:")
    fairness_analysis(models["Random Forest"], X_test, y_test)

