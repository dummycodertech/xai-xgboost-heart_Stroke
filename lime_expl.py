import os
import dill  
import logging
import lime.lime_tabular
import pickle # Still needed to load the training data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LimeExplainerSetup:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.explainer = None
        
    def setup_explainer(self):
        logging.info("Loading training data to initialize LIME...")
        
        # Load the training data using pickle
        with open(os.path.join(self.data_dir, 'train_processed.pkl'), 'rb') as f:
            train_data = pickle.load(f)
            X_train = train_data['X']
            
        feature_names = X_train.columns.tolist()
        
        logging.info("Initializing LIME Tabular Explainer...")
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=['No Stroke', 'Stroke'],
            mode='classification',
            random_state=42
        )
        
        # Save the explainer using DILL instead of pickle
        with open(os.path.join(self.model_dir, 'lime_explainer.pkl'), 'wb') as f:
            dill.dump(self.explainer, f)
            
        logging.info("✅ LIME Explainer initialized and saved successfully using dill.")

if __name__ == "__main__":
    setup = LimeExplainerSetup()
    setup.setup_explainer()