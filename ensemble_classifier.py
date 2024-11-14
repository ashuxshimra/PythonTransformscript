# ensemble_classifier.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging
from tqdm import tqdm
import warnings
import gc
from pathlib import Path

warnings.filterwarnings('ignore')

class AdvancedPowerQualityAnalyzer:
    def __init__(self, sampling_rate=10000, signal_duration=1.0):
        """Initialize the Advanced Power Quality Analyzer"""
        self.sampling_rate = sampling_rate
        self.signal_duration = signal_duration
        self.scaler = StandardScaler()
        
        # Create output directory
        self.output_dir = Path('ensemble_results')
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_enhanced_features(self, signals):
        """Extract comprehensive features with memory efficiency"""
        logging.info("Extracting features...")
        features = []
        
        for signal in tqdm(signals, desc="Processing signals"):
            try:
                # Time domain features
                feature_dict = {
                    'mean': np.mean(signal),
                    'std': np.std(signal),
                    'rms': np.sqrt(np.mean(signal**2)),
                    'peak': np.max(np.abs(signal)),
                    'crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
                    'kurtosis': self._calculate_kurtosis(signal),
                    'skewness': self._calculate_skewness(signal)
                }
                
                # Frequency domain features
                freq_features = self._extract_frequency_features(signal)
                feature_dict.update(freq_features)
                
                # Quality indices
                quality_indices = self._calculate_quality_indices(signal)
                feature_dict.update(quality_indices)
                
                # Convert to list
                features.append(list(feature_dict.values()))
                
                # Clear memory periodically
                if len(features) % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing signal: {str(e)}")
                features.append([0] * 15)  # Add zero features if processing fails
        
        return np.array(features)
    
    def _calculate_kurtosis(self, signal):
        """Calculate kurtosis of the signal"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0
        return np.mean(((signal - mean) / std) ** 4) - 3
    
    def _calculate_skewness(self, signal):
        """Calculate skewness of the signal"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return 0
        return np.mean(((signal - mean) / std) ** 3)
    
    def _extract_frequency_features(self, signal):
        """Extract frequency domain features"""
        try:
            # Calculate FFT
            fft_vals = np.abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)
            
            # Find fundamental frequency
            fundamental_idx = np.argmax(fft_vals[:len(freqs)//2])
            fundamental_freq = freqs[fundamental_idx]
            
            # Calculate spectral features
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * fft_vals[:len(freqs)//2]) / np.sum(fft_vals[:len(freqs)//2])
            spectral_spread = np.sqrt(np.sum(((freqs[:len(freqs)//2] - spectral_centroid)**2) * fft_vals[:len(freqs)//2]) / np.sum(fft_vals[:len(freqs)//2]))
            
            return {
                'fundamental_freq': fundamental_freq,
                'spectral_centroid': spectral_centroid,
                'spectral_spread': spectral_spread,
                'spectral_max': np.max(fft_vals),
                'spectral_mean': np.mean(fft_vals)
            }
            
        except Exception as e:
            logging.error(f"Error in frequency feature extraction: {str(e)}")
            return {
                'fundamental_freq': 0,
                'spectral_centroid': 0,
                'spectral_spread': 0,
                'spectral_max': 0,
                'spectral_mean': 0
            }
    
    def _calculate_quality_indices(self, signal):
        """Calculate power quality indices"""
        try:
            # Calculate RMS value
            rms = np.sqrt(np.mean(signal**2))
            
            # Calculate FFT for THD
            fft_vals = np.abs(np.fft.fft(signal))
            fundamental_idx = np.argmax(fft_vals[:len(signal)//2])
            fundamental = fft_vals[fundamental_idx]
            
            # Get harmonics (up to 7th harmonic)
            harmonics = []
            for i in range(2, 8):
                if fundamental_idx * i < len(fft_vals):
                    harmonics.append(fft_vals[fundamental_idx * i])
            
            # Calculate THD
            thd = np.sqrt(np.sum(np.array(harmonics)**2)) / fundamental * 100 if len(harmonics) > 0 else 0
            
            # Form factor and crest factor
            form_factor = rms / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0
            crest_factor = np.max(np.abs(signal)) / rms if rms != 0 else 0
            
            return {
                'thd': thd,
                'form_factor': form_factor,
                'crest_factor': crest_factor
            }
            
        except Exception as e:
            logging.error(f"Error calculating quality indices: {str(e)}")
            return {
                'thd': 0,
                'form_factor': 0,
                'crest_factor': 0
            }
    
    def build_advanced_model(self):
        """Build optimized ensemble model"""
        try:
            # Use RandomForest with optimized parameters
            model = RandomForestClassifier(
                n_estimators=100,      # Reduced from 100
                max_depth=6,          # Limited depth
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=2,             # Limited parallel jobs
                random_state=42
            )
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            return pipeline
            
        except Exception as e:
            logging.error(f"Error building model: {str(e)}")
            raise

# Only if running as standalone script
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test the analyzer
        analyzer = AdvancedPowerQualityAnalyzer()
        
        # Generate test signal
        t = np.linspace(0, 1.0, 10000)
        test_signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))
        
        # Extract features
        features = analyzer.extract_enhanced_features([test_signal])
        
        logging.info(f"Successfully extracted {features.shape[1]} features")
        
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")