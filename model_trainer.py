import os
import pickle
import json
import logging
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    average_precision_score,
    accuracy_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StrokeModelTrainer:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model = None
        self.metrics = {}
        # NEW: Lowering the clinical threshold from 50% to 15%
        self.decision_threshold = 0.15 
        
    def load_data(self):
        with open(os.path.join(self.data_dir, 'train_processed.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(self.data_dir, 'test_processed.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        return train_data['X'], train_data['y'], test_data['X'], test_data['y']

    def train(self, X_train, y_train):
        logging.info("Training XGBoost with scale_pos_weight...")
        
        # Calculate the ratio of negative to positive cases
        neg_cases = (y_train == 0).sum()
        pos_cases = (y_train == 1).sum()
        scale_weight = neg_cases / pos_cases
        
        logging.info(f"Calculated scale_pos_weight: {scale_weight:.2f}")
        
        self.model = XGBClassifier(
            scale_pos_weight=scale_weight,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(X_train, y_train)
        logging.info("Training complete.")

    def evaluate(self, X_test, y_test):
        logging.info(f"Evaluating model with Custom Threshold: {self.decision_threshold}")
        
        # Get raw probabilities
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Apply custom clinical threshold instead of default 0.50
        y_pred_custom = (y_prob >= self.decision_threshold).astype(int)
        
        acc = accuracy_score(y_test, y_pred_custom)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        logging.info("\n--- CLASSIFICATION REPORT (Threshold = 0.15) ---\n%s", 
                     classification_report(y_test, y_pred_custom))
        
        cm = confusion_matrix(y_test, y_pred_custom)
        logging.info("Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", cm[0][0], cm[0][1], cm[1][0], cm[1][1])
        
        self.metrics = {
            'threshold': self.decision_threshold,
            'accuracy': float(acc),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred_custom, output_dict=True)
        }

    def save_model(self):
        with open(os.path.join(self.model_dir, 'xgboost_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.model_dir, 'training_metadata.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
        logging.info("Model saved.")

if __name__ == "__main__":
    trainer = StrokeModelTrainer()
    X_train, y_train, X_test, y_test = trainer.load_data()
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    trainer.save_model()