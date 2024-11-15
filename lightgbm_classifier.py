import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging
from tqdm import tqdm
import warnings
import gc
from pathlib import Path

warnings.filterwarnings('ignore')

class LightGBMPowerQualityAnalyzer:
    def __init__(self, sampling_rate=10000, signal_duration=1.0):
        """Initialize the LightGBM Power Quality Analyzer"""
        self.sampling_rate = sampling_rate
        self.signal_duration = signal_duration
        self.scaler = StandardScaler()
        
        # Create output directory
        self.output_dir = Path('lightgbm_results')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize feature names
        self.feature_names = None
    
    def extract_enhanced_features(self, signals):
        """Extract comprehensive features optimized for LightGBM"""
        logging.info("Extracting features for LightGBM...")
        features = []
        feature_names = []
        
        for signal in tqdm(signals, desc="Processing signals"):
            try:
                # Time domain features
                time_features = {
                    'mean': np.mean(signal),
                    'std': np.std(signal),
                    'rms': np.sqrt(np.mean(signal**2)),
                    'peak': np.max(np.abs(signal)),
                    'crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
                    'kurtosis': self._calculate_kurtosis(signal),
                    'skewness': self._calculate_skewness(signal),
                    'zero_crossings': np.sum(np.diff(np.signbit(signal).astype(int))),
                    'peak_to_peak': np.max(signal) - np.min(signal)
                }
                
                # Frequency domain features
                fft_vals = np.abs(np.fft.fft(signal))
                freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)
                
                freq_features = {
                    'dominant_freq': freqs[np.argmax(fft_vals[:len(freqs)//2])],
                    'freq_mean': np.mean(fft_vals),
                    'freq_std': np.std(fft_vals),
                    'freq_skewness': self._calculate_skewness(fft_vals),
                    'freq_kurtosis': self._calculate_kurtosis(fft_vals)
                }
                
                # Power quality specific features
                pq_features = self._extract_power_quality_features(signal, fft_vals, freqs)
                
                # Combine all features
                all_features = {**time_features, **freq_features, **pq_features}
                
                # Store feature names only once
                if not feature_names:
                    feature_names = list(all_features.keys())
                
                features.append(list(all_features.values()))
                
                # Memory cleanup
                if len(features) % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing signal: {str(e)}")
                features.append([0] * len(feature_names))
        
        self.feature_names = feature_names
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
    
    def _extract_power_quality_features(self, signal, fft_vals, freqs):
        """Extract power quality specific features"""
        try:
            # Find fundamental frequency component
            fund_idx = np.argmax(fft_vals[:len(freqs)//2])
            fundamental = fft_vals[fund_idx]
            fund_freq = freqs[fund_idx]
            
            # Calculate harmonics
            harmonics = []
            harmonic_ratio = []
            for i in range(2, 8):
                idx = fund_idx * i
                if idx < len(fft_vals):
                    harmonics.append(fft_vals[idx])
                    harmonic_ratio.append(fft_vals[idx] / fundamental)
            
            # THD calculation
            thd = np.sqrt(np.sum(np.array(harmonics)**2)) / fundamental * 100 if harmonics else 0
            
            # Additional power quality metrics
            rms = np.sqrt(np.mean(signal**2))
            form_factor = rms / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) != 0 else 0
            
            features = {
                'thd': thd,
                'form_factor': form_factor,
                'fundamental_freq': fund_freq,
                'fundamental_amplitude': fundamental
            }
            
            # Add individual harmonic ratios
            for i, ratio in enumerate(harmonic_ratio, 2):
                features[f'harmonic_{i}_ratio'] = ratio
            
            return features
            
        except Exception as e:
            logging.error(f"Error in power quality feature extraction: {str(e)}")
            return {
                'thd': 0,
                'form_factor': 0,
                'fundamental_freq': 0,
                'fundamental_amplitude': 0,
                'harmonic_2_ratio': 0,
                'harmonic_3_ratio': 0,
                'harmonic_4_ratio': 0,
                'harmonic_5_ratio': 0,
                'harmonic_6_ratio': 0,
                'harmonic_7_ratio': 0
            }
    
    def build_model(self):
        """Build optimized LightGBM model"""
        try:
            # LightGBM parameters optimized for power quality classification
            lgb_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': 6,
                'min_data_in_leaf': 20,
                'num_iterations': 100
            }
            
            # Create LightGBM classifier
            model = lgb.LGBMClassifier(**lgb_params)
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            return pipeline
            
        except Exception as e:
            logging.error(f"Error building LightGBM model: {str(e)}")
            raise