import os
import urllib.request
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn import set_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
set_config(transform_output="pandas")

class StrokeDataLoader:
    def __init__(self, data_path="data/healthcare-dataset-stroke-data.csv", output_dir="data"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.preprocessor = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)

    def download_data(self):
        if not os.path.exists(self.data_path):
            logging.info("Dataset not found. Downloading...")
            url = "https://raw.githubusercontent.com/karavokyrismichail/Stroke-Prediction---Random-Forest/main/healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv"
            urllib.request.urlretrieve(url, self.data_path)
            logging.info("Download complete.")

    def load_and_preprocess(self):
        self.download_data()
        
        logging.info("Loading raw dataset...")
        df = pd.read_csv(self.data_path)
        
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            
        df = df[df['gender'] != 'Other']

        # --- NEW: FEATURE ENGINEERING ---
        # Force the model to pay attention to compounding lifestyle risks
        logging.info("Engineering new feature: risk_factors_count")
        high_risk_smokers = df['smoking_status'].isin(['smokes', 'formerly smoked']).astype(int)
        df['risk_factors_count'] = df['hypertension'] + df['heart_disease'] + high_risk_smokers

        X = df.drop('stroke', axis=1)
        y = df['stroke']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Added risk_factors_count to numeric features
        numeric_features = ['age', 'avg_glucose_level', 'bmi', 'risk_factors_count']
        categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                                'work_type', 'Residence_type', 'smoking_status']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        logging.info("Scaling and encoding features...")
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # --- REMOVED SMOTE ---
        logging.info("Saving imbalanced datasets and preprocessor...")
        # We now save the raw, imbalanced (but scaled/encoded) data directly
        train_data = {'X': X_train_processed, 'y': y_train}
        test_data = {'X': X_test_processed, 'y': y_test}
        
        with open(os.path.join(self.output_dir, 'train_processed.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
            
        with open(os.path.join(self.output_dir, 'test_processed.pkl'), 'wb') as f:
            pickle.dump(test_data, f)
            
        with open(os.path.join('models', 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(self.preprocessor, f)
            
        logging.info("Data loading and preprocessing complete.")
        return train_data, test_data

if __name__ == "__main__":
    loader = StrokeDataLoader()
    loader.load_and_preprocess()