import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    BaggingClassifier, 
    ExtraTreesClassifier
)
from sklearn.metrics import accuracy_score, precision_score

# ==========================================
# 1. LOAD AND PREP DATA (Fast Mode)
# ==========================================
print("Loading and preparing data...")
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.sample(20000, random_state=42) # Keep it manageable for 9 models
df = df.iloc[:, [0, 1]] 
df.columns = ['target', 'text']
df['target'] = df['target'].map({'spam': 1, 'ham': 0, 1: 1, 0: 0})
df = df.dropna().drop_duplicates()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

df['text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
X_train_vec = tfidf.fit_transform(X_train).toarray()
X_test_vec = tfidf.transform(X_test).toarray()

# ==========================================
# 2. DEFINE THE 9 MODELS
# ==========================================
# We store them in a dictionary to loop through them cleanly
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVC": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(max_depth=50),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Bagging Classifier": BaggingClassifier(n_estimators=50, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=50, random_state=42)
}

# ==========================================
# 3. TRAIN AND EVALUATE
# ==========================================
print("\nTraining 9 models... (This will take a few minutes)")
accuracy_scores = []
precision_scores = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    
    accuracy_scores.append(acc)
    precision_scores.append(prec)

# ==========================================
# 4. CREATE VISUALIZATION FOR PPT
# ==========================================
print("\nGenerating comparison charts...")

# Create a DataFrame for easy plotting
performance_df = pd.DataFrame({
    'Algorithm': list(models.keys()),
    'Accuracy': accuracy_scores,
    'Precision': precision_scores
}).sort_values(by='Precision', ascending=False) # Sort by Precision (most important for spam)

print("\n--- FINAL RANKINGS ---")
print(performance_df.to_string(index=False))

# Draw the bar chart
plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid")

# Melt the dataframe so we can plot Accuracy and Precision side-by-side
melted_df = performance_df.melt(id_vars="Algorithm", var_name="Metric", value_name="Score")

sns.barplot(x="Algorithm", y="Score", hue="Metric", data=melted_df, palette="viridis")
plt.ylim(0.8, 1.0) # Zoom in on the top percentages to see differences clearly
plt.title("Model Comparison: Accuracy vs. Precision", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the graph as an image file
plt.savefig("model_comparison_chart.png", dpi=300)
print("\nSuccess! Check your folder for 'model_comparison_chart.png' to put in your presentation.")