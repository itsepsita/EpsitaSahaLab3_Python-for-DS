from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Fits the model with the training data, makes predictions on test data,
    and returns accuracy, confusion matrix, and classification report.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    return accuracy, cm, cr

def evaluate_all_models(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates various models, returning their performance.
    """
    models = {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'GradientBoost': GradientBoostingClassifier()
    }
    
    results = {}
    for name, model in models.items():
        accuracy, cm, cr = train_and_evaluate(model, X_train, X_test, y_train, y_test)
        results[name] = {'Accuracy': accuracy, 'Confusion Matrix': cm, 'Classification Report': cr}
    
    return results
