
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


class Model:
    def __init__(self, test_size, dataset='data.csv', random_state=42):
        # Store model configuration
        self.test_size = test_size
        self.dataset = dataset
        self.random_state = random_state

        # Load the dataset
        df = pd.read_csv(self.dataset)

        # Select input features
        self.X = df[[
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se',
            'smoothness_se', 'compactness_se', 'concavity_se',
            'concave points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
            'smoothness_worst', 'compactness_worst', 'concavity_worst',
            'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst',
        ]]

        # Define target labels
        self.y = df["diagnosis"]

        # Initialize the classification model and scaler
        self.model = LogisticRegression(max_iter=10000)
        self.scaler = StandardScaler()

    def split_data(self):
        # Split dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # Scale features using training data statistics
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        # Train the model using the training dataset
        self.model.fit(self.X_train, self.y_train)

    def predict(self, df, threshold=0.5):
        # Scale new input data
        df_scaled = self.scaler.transform(df)

        # Calculate class probabilities
        probabilities = self.model.predict_proba(df_scaled)
        prob_cancer = probabilities[0][1]

        # Apply classification threshold
        prediction = 1 if prob_cancer >= threshold else 0

        return {
            "prediction": prediction,
            "prob_cancer": prob_cancer
        }

    def evaluate_model(self):
        # Generate predictions on the test dataset
        y_pred = self.model.predict(self.X_test)

        # Calculate evaluation metrics
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        return {
            "acc": acc,
            "report": report
        }
