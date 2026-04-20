import pandas as pd
import numpy as np
import pickle

# Machine Learning Modules
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score

# ==========================================
# 1. LOAD AND CLEAN THE DATA
# ==========================================
print("Loading data...")
# Read the CSV file. We use latin-1 encoding because emails often have weird characters
df = pd.read_csv('data/spam.csv', encoding='latin-1')

#Random 20000 samples
df=df.sample(20000,random_state=42)

# Kaggle datasets often have extra empty columns at the end, let's drop them
df = df.iloc[:, [0, 1]] 

# Rename columns so they are easy to work with
df.columns = ['target', 'text']

# Convert the 'target' column to numbers: 'spam' = 1, 'ham' (not spam) = 0
# If your dataset uses 1 and 0 already, this just leaves it as is.
df['target'] = df['target'].map({'spam': 1, 'ham': 0, 1: 1, 0: 0})

# Drop any duplicate emails or missing values
df = df.drop_duplicates(keep='first')
df = df.dropna()

# ==========================================
# 2. SPLIT AND VECTORIZE
# ==========================================
# Split data: 80% for training the model, 20% for testing its accuracy
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

print("Vectorizing text...")
# TF-IDF converts the text into numbers based on how important words are
# max_features=3000 means we only look at the top 3000 most common words to save memory
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')

X_train_vectorized = tfidf.fit_transform(X_train).toarray()
X_test_vectorized = tfidf.transform(X_test).toarray()

# ==========================================
# 3. TRAIN INDIVIDUAL MODELS (FOR YOUR PPT)
# ==========================================
print("\n--- Training Individual Models ---")

# Model A: Naive Bayes (Your old model baseline)
nb = MultinomialNB()
nb.fit(X_train_vectorized, y_train)
y_pred_nb = nb.predict(X_test_vectorized)
print(f"Naive Bayes -> Accuracy: {accuracy_score(y_test, y_pred_nb):.4f} | Precision: {precision_score(y_test, y_pred_nb):.4f}")

# Model B: Support Vector Classifier (The strong text classifier)
# probability=True is required so it can participate in the Voting Classifier later
svc = SVC(kernel='linear', probability=True) 
svc.fit(X_train_vectorized, y_train)
y_pred_svc = svc.predict(X_test_vectorized)
print(f"SVC         -> Accuracy: {accuracy_score(y_test, y_pred_svc):.4f} | Precision: {precision_score(y_test, y_pred_svc):.4f}")

# Model C: Random Forest (Tree-based)
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train_vectorized, y_train)
y_pred_rf = rf.predict(X_test_vectorized)
print(f"Random Tree -> Accuracy: {accuracy_score(y_test, y_pred_rf):.4f} | Precision: {precision_score(y_test, y_pred_rf):.4f}")


# ==========================================
# 4. THE VOTING CLASSIFIER (YOUR FINAL MODEL)
# ==========================================
print("\n--- Training Final Voting Classifier ---")
# 'soft' voting means it averages the probability scores of the 3 models for the final decision
voting_clf = VotingClassifier(
    estimators=[('nb', nb), ('svc', svc), ('rf', rf)], 
    voting='soft'
)

voting_clf.fit(X_train_vectorized, y_train)
y_pred_voting = voting_clf.predict(X_test_vectorized)
print(f"VOTING CLF  -> Accuracy: {accuracy_score(y_test, y_pred_voting):.4f} | Precision: {precision_score(y_test, y_pred_voting):.4f}")


# ==========================================
# 5. SAVE THE MODEL FOR DEPLOYMENT
# ==========================================
print("\nSaving model and vectorizer...")
# We must save BOTH the tfidf and the voting_clf to use them in FastAPI
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(voting_clf, open('model.pkl', 'wb'))

print("Done! Files saved as vectorizer.pkl and model.pkl")