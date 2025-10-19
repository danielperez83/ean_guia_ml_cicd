import os, joblib, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

os.makedirs("Results", exist_ok=True)
model = joblib.load("Model/model.pkl") if os.path.exists("Model/model.pkl") else None

X, y = make_classification(
    n_samples=400,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=123,
)

yp = model.predict(X) if model else np.zeros_like(y)
acc = accuracy_score(y, yp)
f1w = f1_score(y, yp, average="weighted")
cm = confusion_matrix(y, yp)

with open("Results/metrics.txt", "w") as f:
    f.write(f"accuracy: {acc:.4f}\n")
    f.write(f"f1_weighted: {f1w:.4f}\n")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.title("Confusion Matrix (eval)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("Results/model_results.png")
plt.close()
print("Evaluation complete → Results/metrics.txt & Results/model_results.png")
