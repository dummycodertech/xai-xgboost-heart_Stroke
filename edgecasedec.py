import os
import pickle
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EdgeCaseDetector:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.edge_cases = {}
        
    def load_artifacts(self):
        logging.info("Loading test data and model...")
        with open(os.path.join(self.model_dir, 'xgboost_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
            
        with open(os.path.join(self.data_dir, 'test_processed.pkl'), 'rb') as f:
            test_data = pickle.load(f)
            self.X_test = test_data['X']
            self.y_test = test_data['y'].values # Convert to numpy for easy indexing

    def find_cases(self):
        logging.info("Scanning for edge cases in the test set...")
        
        # Get model predictions and probabilities
        y_pred = self.model.predict(self.X_test)
        y_probs = self.model.predict_proba(self.X_test)[:, 1]
        
        # 1. False Positives: Model predicted stroke (1), actual is no stroke (0)
        fp_mask = (y_pred == 1) & (self.y_test == 0)
        fp_indices = np.where(fp_mask)[0]
        
        # 2. False Negatives: Model predicted no stroke (0), actual is stroke (1)
        fn_mask = (y_pred == 0) & (self.y_test == 1)
        fn_indices = np.where(fn_mask)[0]
        
        # 3. Uncertain Boundaries: Model is 45-55% confident (basically a coin toss)
        uncertain_mask = (y_probs >= 0.45) & (y_probs <= 0.55)
        uncertain_indices = np.where(uncertain_mask)[0]
        
        # Store up to 5 examples of each for the dashboard
        # We store the index, probability, and actual status
        self.edge_cases = {
            "false_positives": [
                {"index": int(idx), "prob": float(y_probs[idx]), "actual": 0} 
                for idx in fp_indices[:5]
            ],
            "false_negatives": [
                {"index": int(idx), "prob": float(y_probs[idx]), "actual": 1} 
                for idx in fn_indices[:5]
            ],
            "uncertain": [
                {"index": int(idx), "prob": float(y_probs[idx]), "actual": int(self.y_test[idx])} 
                for idx in uncertain_indices[:5]
            ]
        }
        
        logging.info("Found %d False Positives, %d False Negatives, %d Uncertain Cases.", 
                     len(fp_indices), len(fn_indices), len(uncertain_indices))

    def save_cases(self):
        logging.info("Saving edge cases for the dashboard...")
        with open(os.path.join(self.model_dir, 'edge_cases.json'), 'w') as f:
            json.dump(self.edge_cases, f, indent=4)
        logging.info("Edge cases saved successfully.")

if __name__ == "__main__":
    detector = EdgeCaseDetector()
    detector.load_artifacts()
    detector.find_cases()
    detector.save_cases()