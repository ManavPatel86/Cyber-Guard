import pickle
import joblib

print("Loading the massive 131MB model...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Compressing model...")
# compress=3 is the sweet spot for maximum shrinkage without slowing down the API
joblib.dump(model, 'model.joblib', compress=3)

print("Done! You can now safely delete the giant model.pkl file.")