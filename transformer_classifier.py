# transformer_classifier.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings('ignore')

class PowerQualityDataset(Dataset):
    def __init__(self, signals, labels):
        if len(signals.shape) == 2:
            signals = signals.reshape(signals.shape[0], -1, 1)
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

class PowerQualityTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=16, nhead=2, num_layers=1, num_classes=3, dropout=0.1):
        super().__init__()
        
        # Input embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, d_model)
        x = self.transformer(x)  # Shape: (batch_size, sequence_length, d_model)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.decoder(x)  # Shape: (batch_size, num_classes)
        return x

class PowerQualityAnalysis:
    def __init__(self, sampling_rate=10000, signal_duration=1.0):
        """Initialize the power quality analysis framework"""
        self.sampling_rate = sampling_rate
        self.signal_duration = signal_duration
        self.device = torch.device('cpu')  # Force CPU usage for stability
        self.scaler = StandardScaler()
        self.history = {
            'train_losses': [],
            'test_losses': [],
            'train_accuracies': [],
            'test_accuracies': []
        }

    def generate_synthetic_dataset(self, n_samples=10):
        """Generate synthetic dataset with memory efficiency"""
        logging.info(f"Generating {n_samples} synthetic signals...")
        t = np.linspace(0, self.signal_duration, int(self.sampling_rate * self.signal_duration))
        signals = []
        labels = []
        frequency = 50  # Base frequency (50 Hz)
        
        for _ in tqdm(range(n_samples), desc="Generating signals"):
            try:
                signal_type = np.random.randint(0, 3)
                signal = np.sin(2 * np.pi * frequency * t)
                
                if signal_type == 0:  # High quality
                    noise = np.random.normal(0, 0.01, len(t))
                    signal = signal + noise
                    
                elif signal_type == 1:  # Medium quality (harmonics)
                    third_harmonic = 0.15 * np.sin(2 * np.pi * 3 * frequency * t)
                    fifth_harmonic = 0.1 * np.sin(2 * np.pi * 5 * frequency * t)
                    signal = signal + third_harmonic + fifth_harmonic
                    
                else:  # Low quality (dips and transients)
                    dip_start = np.random.randint(1000, 8000)
                    dip_duration = np.random.randint(500, 1500)
                    signal[dip_start:dip_start+dip_duration] *= 0.7
                    
                    transient_start = np.random.randint(1000, 8000)
                    transient_duration = 50
                    signal[transient_start:transient_start+transient_duration] *= 1.5
                
                signals.append(signal)
                labels.append(signal_type)
                
            except Exception as e:
                logging.error(f"Error generating signal: {str(e)}")
                continue
        
        signals = np.array(signals)
        labels = np.array(labels)
        
        return signals, labels
    
    def prepare_data(self, signals, labels, batch_size=4):
        """Prepare data with memory-efficient settings"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                signals, labels, test_size=0.2, random_state=42
            )
            
            # Scale data
            X_train_reshaped = X_train.reshape(-1, 1)
            X_test_reshaped = X_test.reshape(-1, 1)
            
            # Fit scaler on training data only
            self.scaler.fit(X_train_reshaped)
            
            # Transform both datasets
            X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
            X_test_scaled = self.scaler.transform(X_test_reshaped).reshape(X_test.shape)
            
            # Create datasets
            train_dataset = PowerQualityDataset(X_train_scaled, y_train)
            test_dataset = PowerQualityDataset(X_test_scaled, y_test)
            
            # Create dataloaders with memory-efficient settings
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            return train_loader, test_loader, X_test_scaled, y_test
            
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise
    
    def train_transformer(self, train_loader, test_loader, epochs=5):
        """Train transformer with enhanced monitoring and memory management"""
        try:
            # Initialize model and training components
            model = PowerQualityTransformer().to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=0.001,
                weight_decay=0.01
            )
            
            # Reset history
            self.history = {
                'train_losses': [],
                'test_losses': [],
                'train_accuracies': [],
                'test_accuracies': []
            }
            
            # Training loop
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
                for batch_idx, (signals, labels) in enumerate(train_pbar):
                    try:
                        signals = signals.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Forward pass
                        optimizer.zero_grad()
                        outputs = model(signals)
                        loss = criterion(outputs, labels)
                        
                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        # Calculate metrics
                        train_loss += loss.item()
                        _, predicted = outputs.max(1)
                        train_total += labels.size(0)
                        train_correct += predicted.eq(labels).sum().item()
                        
                        # Update progress bar
                        train_accuracy = 100. * train_correct / train_total
                        train_pbar.set_postfix({
                            'loss': f'{train_loss/(batch_idx+1):.4f}',
                            'acc': f'{train_accuracy:.2f}%'
                        })
                        
                        # Clear memory
                        del signals, labels, outputs, loss
                        gc.collect()
                        
                    except RuntimeError as e:
                        logging.error(f"Training error in batch {batch_idx}: {str(e)}")
                        continue
                
                # Validation phase
                model.eval()
                test_loss = 0
                test_correct = 0
                test_total = 0
                
                with torch.no_grad():
                    test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Test]')
                    for batch_idx, (signals, labels) in enumerate(test_pbar):
                        try:
                            signals = signals.to(self.device)
                            labels = labels.to(self.device)
                            
                            # Forward pass
                            outputs = model(signals)
                            loss = criterion(outputs, labels)
                            
                            # Calculate metrics
                            test_loss += loss.item()
                            _, predicted = outputs.max(1)
                            test_total += labels.size(0)
                            test_correct += predicted.eq(labels).sum().item()
                            
                            # Update progress bar
                            test_accuracy = 100. * test_correct / test_total
                            test_pbar.set_postfix({
                                'loss': f'{test_loss/(batch_idx+1):.4f}',
                                'acc': f'{test_accuracy:.2f}%'
                            })
                            
                            # Clear memory
                            del signals, labels, outputs, loss
                            gc.collect()
                            
                        except RuntimeError as e:
                            logging.error(f"Validation error in batch {batch_idx}: {str(e)}")
                            continue
                
                # Calculate epoch metrics
                train_loss = train_loss / len(train_loader)
                test_loss = test_loss / len(test_loader)
                train_accuracy = 100. * train_correct / train_total
                test_accuracy = 100. * test_correct / test_total
                
                # Update history
                self.history['train_losses'].append(train_loss)
                self.history['test_losses'].append(test_loss)
                self.history['train_accuracies'].append(train_accuracy)
                self.history['test_accuracies'].append(test_accuracy)
                
                # Log progress
                logging.info(f'Epoch {epoch+1}/{epochs}:')
                logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
                logging.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
                
                # Clear memory between epochs
                gc.collect()
            
            return model, self.history
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise