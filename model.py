import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

os.makedirs("Results", exist_ok=True)
os.makedirs("Model", exist_ok=True)

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=42,
)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline(
    [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
)
pipe.fit(Xtr, ytr)
yp = pipe.predict(Xte)

acc = accuracy_score(yte, yp)
f1w = f1_score(yte, yp, average="weighted")
cm = confusion_matrix(yte, yp)

with open("Results/metrics.txt", "w") as f:
    f.write(f"accuracy: {acc:.4f}\n")
    f.write(f"f1_weighted: {f1w:.4f}\n")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("Results/model_results.png")
plt.close()

joblib.dump(pipe, "Model/model.pkl")
print("Training complete → Results/metrics.txt & Results/model_results.png")
