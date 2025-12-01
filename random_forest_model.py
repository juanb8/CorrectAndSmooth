import numpy as np
from pandas.core.common import random_state
import torch
from sklearn.ensemble import RandomForestClassifier
import joblib

class RandomForestWrapper:
    def __init__(self, n_estimators=100, random_state=42, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            **kwargs
        )
        self.n_estimators = n_estimators
    
    def fit(self, data, split_idx):
        # Convert to numpy (same format as other models expect)
        features = data.x.numpy()
        labels = data.y.numpy().ravel()
        train_idx = split_idx['train'].numpy()
        
        self.model.fit(features[train_idx], labels[train_idx])
        return self  # for chaining
    
    def forward(self, data):
        """Mimic PyTorch model interface - outputs probabilities"""
        return self.predict_proba(data)
    
    def predict_proba(self, data):
        features = data.x.numpy()
        proba = self.model.predict_proba(features)
        return torch.from_numpy(proba).float()
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path):
        wrapper = cls()
        wrapper.model = joblib.load(path)
        return wrapper
    
    def eval(self):
        """PyTorch compatibility method"""
        return self
    
    def to(self, device):
        """PyTorch compatibility method - RF doesn't use GPU"""
        return self

    def parameters(self):
        """PyTorch compatibility method - return empty iterator"""
        return iter([])
    
    def num_parameters(self):
        """Get approximate parameter count"""
        if hasattr(self.model, 'estimators_'):
            # Rough estimate: n_trees * n_features * max_depth
            return self.n_estimators * (self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 100)
        return 0

