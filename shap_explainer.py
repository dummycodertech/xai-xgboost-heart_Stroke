import os
import pickle
import logging
import shap
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ShapExplainerSetup:
    def __init__(self, model_dir="models", data_dir="data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.X_test = None
        self.explainer = None
        self.shap_values = None

    def load_artifacts(self):
        logging.info("Loading trained XGBoost model and test data...")
        with open(os.path.join(self.model_dir, 'xgboost_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
            
        with open(os.path.join(self.data_dir, 'test_processed.pkl'), 'rb') as f:
            test_data = pickle.load(f)
            self.X_test = test_data['X']

    def generate_global_explanations(self):
        logging.info("Initializing SHAP TreeExplainer...")
        # TreeExplainer is highly optimized specifically for tree-based models like XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        
        logging.info("Calculating SHAP values for the test set (this might take a few seconds)...")
        # Calling the explainer directly generates a shap.Explanation object, 
        # which is required for modern SHAP plots like waterfall and beeswarm.
        self.shap_values = self.explainer(self.X_test)
        
        logging.info("✅ SHAP calculations complete.")

    def save_artifacts(self):
        logging.info("Saving SHAP explainer and pre-computed values for Streamlit...")
        
        # Save the explainer for Tab 1 (to explain a single user prediction on the fly)
        with open(os.path.join(self.model_dir, 'shap_explainer.pkl'), 'wb') as f:
            pickle.dump(self.explainer, f)
            
        # Save the pre-calculated test set explanations for Tab 2 (instant global plots)
        with open(os.path.join(self.model_dir, 'shap_values_test.pkl'), 'wb') as f:
            pickle.dump(self.shap_values, f)
            
        logging.info("✅ SHAP artifacts successfully saved.")

    def test_plots(self):
        # This verifies the plots generate correctly during pipeline execution.
        # Streamlit will use these same commands internally later.
        logging.info("Testing global plot generation...")
        
        os.makedirs("plots", exist_ok=True)
        
        # Test Beeswarm Plot (Tab 2)
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(self.shap_values, show=False)
        plt.tight_layout()
        plt.savefig("plots/shap_summary_beeswarm.png")
        plt.close()
        
        # Test Bar Plot (Tab 2)
        plt.figure(figsize=(10, 6))
        shap.plots.bar(self.shap_values, show=False)
        plt.tight_layout()
        plt.savefig("plots/shap_summary_bar.png")
        plt.close()
        
        logging.info("✅ Test plots saved to plots/ directory.")

if __name__ == "__main__":
    setup = ShapExplainerSetup()
    setup.load_artifacts()
    setup.generate_global_explanations()
    setup.save_artifacts()
    setup.test_plots()