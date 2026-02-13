"""
Deep learning models for tabular data benchmark.
Includes MLP and TabNet.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MLPModel(nn.Module):
    """Simple Multi-Layer Perceptron for tabular data."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64], 
                 output_dim: int = 2, dropout: float = 0.2):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


class MLP:
    """
    MLP wrapper with sklearn-like API.
    """
    
    def __init__(self, task_type: str = 'classification', 
                 hidden_dims: list = [128, 64],
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 epochs: int = 100,
                 early_stopping_patience: int = 10,
                 device: str = None):
        
        self.task_type = task_type
        self.is_classifier = task_type == 'classification'
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.classes_ = None
        
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = False):
        """Prepare data for training/inference."""
        # Handle categorical features
        X_processed = X.copy()
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            X_processed[col] = X_processed[col].astype('category').cat.codes
        
        X_numeric = X_processed.select_dtypes(include=[np.number]).values
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_scaled = self.scaler.transform(X_numeric)
        
        if y is not None:
            if self.is_classifier:
                y_array = y.values if isinstance(y, pd.Series) else y
            else:
                y_array = y.values if isinstance(y, pd.Series) else y
            return torch.FloatTensor(X_scaled), torch.LongTensor(y_array) if self.is_classifier else torch.FloatTensor(y_array)
        
        return torch.FloatTensor(X_scaled)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train the MLP model."""
        
        # Store classes for classification
        if self.is_classifier:
            self.classes_ = np.unique(y_train)
            self.n_classes = len(self.classes_)
        
        # Prepare data
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train, fit=True)
        
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val, fit=False)
        else:
            # Use last 20% as validation
            val_size = int(0.2 * len(X_train_tensor))
            X_val_tensor = X_train_tensor[-val_size:]
            y_val_tensor = y_train_tensor[-val_size:]
            X_train_tensor = X_train_tensor[:-val_size]
            y_train_tensor = y_train_tensor[:-val_size]
        
        # Create model
        input_dim = X_train_tensor.shape[1]
        output_dim = self.n_classes if self.is_classifier else 1
        
        self.model = MLPModel(input_dim, self.hidden_dims, output_dim, self.dropout).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        if self.is_classifier:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                if self.is_classifier:
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                X_val_tensor = X_val_tensor.to(self.device)
                y_val_tensor = y_val_tensor.to(self.device)
                val_outputs = self.model(X_val_tensor)
                
                if self.is_classifier:
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                else:
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_tensor = self._prepare_data(X_test, fit=False).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.is_classifier:
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = outputs.squeeze().cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classification")
        
        X_tensor = self._prepare_data(X_test, fit=False).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probabilities
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        metrics = {}
        
        if self.is_classifier:
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            if len(self.classes_) == 2:
                try:
                    proba = self.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, proba)
                except:
                    pass
        else:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
            metrics['r2'] = r2_score(y_test, predictions)
        
        return metrics


class TabNetModel:
    """
    TabNet model wrapper.
    TabNet uses sequential attention to choose which features to reason from at each decision step.
    """
    
    def __init__(self, task_type: str = 'classification',
                 n_d: int = 8, n_a: int = 8, n_steps: int = 3,
                 gamma: float = 1.3, lambda_sparse: float = 1e-4,
                 max_epochs: int = 100, patience: int = 10,
                 batch_size: int = 256):
        
        self.task_type = task_type
        self.is_classifier = task_type == 'classification'
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = StandardScaler()
        self.classes_ = None
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = False):
        """Prepare data for TabNet."""
        X_processed = X.copy()
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            X_processed[col] = X_processed[col].astype('category').cat.codes
        
        X_numeric = X_processed.select_dtypes(include=[np.number]).values
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_scaled = self.scaler.transform(X_numeric)
        
        if y is not None:
            y_array = y.values if isinstance(y, pd.Series) else y
            return X_scaled, y_array
        
        return X_scaled
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train TabNet model."""
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        except ImportError:
            print("TabNet not installed. Please run: pip install pytorch-tabnet")
            raise
        
        # Store classes
        if self.is_classifier:
            self.classes_ = np.unique(y_train)
        
        # Prepare data
        X_train_processed, y_train_processed = self._prepare_data(X_train, y_train, fit=True)
        
        if X_val is not None and y_val is not None:
            X_val_processed, y_val_processed = self._prepare_data(X_val, y_val, fit=False)
        else:
            # Use last 20% as validation
            val_size = int(0.2 * len(X_train_processed))
            X_val_processed = X_train_processed[-val_size:]
            y_val_processed = y_train_processed[-val_size:]
            X_train_processed = X_train_processed[:-val_size]
            y_train_processed = y_train_processed[:-val_size]
        
        # Create model
        if self.is_classifier:
            self.model = TabNetClassifier(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                lambda_sparse=self.lambda_sparse,
                verbose=0,
                seed=42
            )
            
            self.model.fit(
                X_train=X_train_processed,
                y_train=y_train_processed,
                eval_set=[(X_val_processed, y_val_processed)],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size
            )
        else:
            self.model = TabNetRegressor(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                lambda_sparse=self.lambda_sparse,
                verbose=0,
                seed=42
            )
            
            self.model.fit(
                X_train=X_train_processed,
                y_train=y_train_processed.reshape(-1, 1),
                eval_set=[(X_val_processed, y_val_processed.reshape(-1, 1))],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size
            )
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_processed = self._prepare_data(X_test, fit=False)
        predictions = self.model.predict(X_processed)
        
        if self.is_classifier:
            return predictions
        else:
            return predictions.flatten()
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classification")
        
        X_processed = self._prepare_data(X_test, fit=False)
        return self.model.predict_proba(X_processed)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        metrics = {}
        
        if self.is_classifier:
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            if len(self.classes_) == 2:
                try:
                    proba = self.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, proba)
                except:
                    pass
        else:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
            metrics['r2'] = r2_score(y_test, predictions)
        
        return metrics


if __name__ == "__main__":
    # Test deep learning models
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    print("Testing MLP...")
    mlp = MLP('classification', epochs=10)
    mlp.fit(X_train, y_train)
    metrics = mlp.evaluate(X_test, y_test)
    print(f"MLP Metrics: {metrics}")
