import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load processed data from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def prepare_data(data):
    """Prepare features and labels from the loaded data."""
    X = [np.array(run["Functional"][0:4]).flatten() for participant in data for run in participant]
    y = [run["Shape"] for participant in data for run in participant]
    return X, y

def main():
    # Load the data
    all_parts = load_data('processed_data.pickle')

    # Prepare the data
    X, y = prepare_data(all_parts)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define the pipeline for preprocessing and classification
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.9)),
        ('classifier', SVC(kernel='linear', class_weight='balanced'))
    ])

    # Train the classifier
    print("Training the classifier...")
    pipeline.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()


