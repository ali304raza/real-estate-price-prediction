import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.inspection import permutation_importance


from Data import(load_data, prepare_data, build_preprocessing_pipeline)

df = load_data()
X_train, X_test, y_train, y_test = prepare_data(df)

Pipeline = Pipeline([("preprocessor", build_preprocessing_pipeline(X_train)), ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42))])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = GridSearchCV(Pipeline, {"classifier__C": [0.1, 1.0, 10.0]}, cv=cv, scoring="f1_macro", n_jobs=-1)

print("\nTraining model ....")
search.fit(X_train, y_train)

model = search.best_estimator_

print("Best parameters:", search.best_params_)
print("Best Cv macro F1", round(search.best_score_, 4))

y_pred = model.predict(X_test)

print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nTest Macro F1:", round(f1_score(y_test, y_pred, average="macro"), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nCalculating feature importance...")

result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1_macro",
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)


importances = pd.DataFrame({
    "feature": X_test.columns,
    "importance": result.importances_mean
}).sort_values("importance", ascending=False)

print("\nTop 5 Important Features:")
print(importances.head(5).to_string(index=False))


os.makedirs("artifacts", exist_ok=True)

joblib.dump(model, "artifacts/model.joblib")
print("\nModel saved successfully")
