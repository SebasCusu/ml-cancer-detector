from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class Model:
    def __init__(self, test_size, dataset='data.csv', random_state=42):
        self.test_size = test_size
        self.dataset = dataset
        self.random_state = random_state
        df = pd.read_csv(self.dataset)
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
        self.y = df["diagnosis"]
        self.model = LogisticRegression(max_iter=10000)
        self.scaler = StandardScaler()
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
    
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self, df, threshold=0.5):
        df_scaled = self.scaler.transform(df)
        probabilities = self.model.predict_proba(df_scaled)
        prob_cancer = probabilities[0][1]
        prediction = 1 if prob_cancer >= threshold else 0
        
        return {
            "prediction" : prediction,
            "prob_cancer" : prob_cancer
        }
    
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        return {
            "acc" : acc,
            "report" : report
        }
        

data = {
    'radius_mean': [11.42],
    'texture_mean': [20.38],
    'perimeter_mean': [77.58],
    'area_mean': [386.1],
    'smoothness_mean': [0.1425],
    'compactness_mean': [0.1489],
    'concavity_mean': [0.05333],
    'concave points_mean': [0.02873],
    'symmetry_mean': [0.2101],
    'fractal_dimension_mean': [0.07687],
    'radius_se': [0.3345],
    'texture_se': [0.8902],
    'perimeter_se': [2.256],
    'area_se': [24.15],
    'smoothness_se': [0.00753],
    'compactness_se': [0.02345],
    'concavity_se': [0.02589],
    'concave points_se': [0.01234],
    'symmetry_se': [0.02123],
    'fractal_dimension_se': [0.00345],
    'radius_worst': [12.34],
    'texture_worst': [26.50],
    'perimeter_worst': [84.00],
    'area_worst': [461.3],
    'smoothness_worst': [0.1700],
    'compactness_worst': [0.3456],
    'concavity_worst': [0.2345],
    'concave points_worst': [0.0987],
    'symmetry_worst': [0.3214],
    'fractal_dimension_worst': [0.0987]
}

df = pd.DataFrame(data)