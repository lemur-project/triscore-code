import torch
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, loss_weights=None):
        super(MLP, self).__init__()
        # save hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(hidden_dim, num_classes)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_weights = loss_weights.to(self.device) if loss_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=self.loss_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.to(self.device)
        
    def fit(self, train_loader, val_loader, epochs, debug=False, early_stopping=False, patience=5):
        print(f"Training projection network...")

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        train_loss_hist = []
        val_loss_hist = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                embeddings = batch['embedding'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
            
            avg_train_loss = epoch_train_loss / num_train_batches
            train_loss_hist.append(avg_train_loss)
            
            # Validation phase
            val_loss, val_f1 = self._validate(val_loader)
            val_loss_hist.append(val_loss)
            
            if debug:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if debug:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
        
        # Restore best model if early stopping was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print("Training complete!")
        return train_loss_hist, val_loss_hist
    
    def _validate(self, val_loader):

        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embedding'].to(self.device)
                labels = batch['label']
                
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
                labels = labels.to(self.device)
                
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
        avg_loss = total_loss / num_batches
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, f1
    
    def predict(self, test_data):

        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            if isinstance(test_data, torch.utils.data.DataLoader):
                for batch in test_data:
                    embeddings = batch['embedding'].to(self.device)
                    outputs = self.model(embeddings)
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
            elif isinstance(test_data, np.ndarray):
                embeddings = torch.tensor(test_data, dtype=torch.float32).to(self.device)
                outputs = self.model(embeddings)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                
        return np.array(all_predictions)
    
    def get_embeddings(self, data):
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            # if the data is a loader
            if isinstance(data, torch.utils.data.DataLoader):
                for batch in data:
                    embeddings = batch['embedding'].to(self.device)
                    # Get output from all layers except the last one (classification layer)
                    hidden_embeddings = self.model[:-1](embeddings)  # Excludes final Linear layer
                    all_embeddings.append(hidden_embeddings.cpu().numpy())
            # if the data is a numpy array
            elif isinstance(data, np.ndarray):
                embeddings = torch.tensor(data, dtype=torch.float32).to(self.device)
                hidden_embeddings = self.model[:-1](embeddings)  # Excludes final Linear layer
                all_embeddings.append(hidden_embeddings.cpu().numpy())
            else:
                raise ValueError("Unsupported data type for embeddings extraction. Use DataLoader or numpy array.")
            
        # Concatenate all batch embeddings
        return np.concatenate(all_embeddings, axis=0)
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            "args": {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_classes': self.num_classes,
                'loss_weights': self.loss_weights
            }
        }, path)

        return path