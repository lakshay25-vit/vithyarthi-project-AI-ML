import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

np.random.seed(42)
n = 1000

hours = np.random.randint(0, 24, n)
day = np.random.randint(0, 7, n)
is_weekend = (day >= 5).astype(int)
rain = np.random.choice([0, 1], n, p=[0.7, 0.3])
temperature = np.random.uniform(15, 42, n)

def get_label(h, weekend, r):
    score = 0
    if 8 <= h <= 10 or 17 <= h <= 19:
        score += 3
    if weekend:
        score -= 1
    if r:
        score += 1
    if score <= 1:
        return "Low"
    elif score == 2:
        return "Moderate"
    else:
        return "Heavy"

labels = [get_label(hours[i], is_weekend[i], rain[i]) for i in range(n)]

df = pd.DataFrame({
    "hour": hours,
    "day": day,
    "is_weekend": is_weekend,
    "rain": rain,
    "temperature": temperature,
    "congestion": labels
})

print(df.head(10))
print()
print(df["congestion"].value_counts())


plt.figure(figsize=(8, 4))
sns.countplot(x="congestion", data=df, palette="RdYlGn_r", order=["Low", "Moderate", "Heavy"])
plt.title("Congestion Level Distribution")
plt.xlabel("Congestion Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("congestion_distribution.png")
plt.show()

plt.figure(figsize=(10, 4))
sns.countplot(x="hour", hue="congestion", data=df, palette="RdYlGn_r",
              hue_order=["Low", "Moderate", "Heavy"])
plt.title("Congestion by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("congestion_by_hour.png")
plt.show()


le = LabelEncoder()
df["label"] = le.fit_transform(df["congestion"])

X = df[["hour", "day", "is_weekend", "rain", "temperature"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_sc, y_train)
lr_pred = lr_model.predict(X_test_sc)
lr_acc = accuracy_score(y_test, lr_pred)

dt_model = DecisionTreeClassifier(max_depth=4)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("Logistic Regression Accuracy:", round(lr_acc * 100, 2), "%")
print("Decision Tree Accuracy:", round(dt_acc * 100, 2), "%")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(confusion_matrix(y_test, lr_pred),
                       display_labels=le.classes_).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay(confusion_matrix(y_test, dt_pred),
                       display_labels=le.classes_).plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Decision Tree")

plt.suptitle("Confusion Matrices", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

print()
print("Classification Report - Decision Tree")
print(classification_report(y_test, dt_pred, target_names=le.classes_))


train_acc = []
test_acc = []
depths = range(1, 12)

for d in depths:
    m = DecisionTreeClassifier(max_depth=d)
    m.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, m.predict(X_train)))
    test_acc.append(accuracy_score(y_test, m.predict(X_test)))

plt.figure(figsize=(8, 4))
plt.plot(depths, train_acc, label="Train Accuracy", marker="o")
plt.plot(depths, test_acc, label="Test Accuracy", marker="s")
plt.title("Overfitting Check - Decision Tree")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("overfitting_check.png")
plt.show()


sample = pd.DataFrame([{
    "hour": 9,
    "day": 0,
    "is_weekend": 0,
    "rain": 1,
    "temperature": 30
}])

result = le.inverse_transform(dt_model.predict(sample))[0]
print("Predicted congestion for Monday 9AM with rain:", result)
