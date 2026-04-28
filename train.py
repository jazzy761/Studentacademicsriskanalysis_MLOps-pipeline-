import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# load
df = pd.read_csv("data.csv")

X = df.drop(columns=['Target'])
y = df['Target']

# encode target
le = LabelEncoder()
y = le.fit_transform(y)

# one hot (same as your notebook)
X = pd.get_dummies(X, columns=['Course'], drop_first=True)

# split BEFORE pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# pipeline (THIS IS THE KEY)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE()),
    ("model", RandomForestClassifier(random_state=42))
])

# grid search
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5]
}

grid = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# eval
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save FULL pipeline
joblib.dump(best_model, "model.pkl")

print("pipeline saved ✅")