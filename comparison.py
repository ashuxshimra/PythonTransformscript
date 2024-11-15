# comparison.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
import logging
import warnings
import json
from datetime import datetime
from pathlib import Path
import gc
import psutil
import sys
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from lightgbm_classifier import LightGBMPowerQualityAnalyzer

from ensemble_classifier import AdvancedPowerQualityAnalyzer
from transformer_classifier import PowerQualityAnalysis

# Suppress warnings
warnings.filterwarnings('ignore')

class PowerQualityComparison:
    def __init__(self):
        """Initialize the comparison framework with optimized settings"""
        self.lightgbm_analyzer = LightGBMPowerQualityAnalyzer()
        self.base_tariff = 1.0  # Base electricity rate
        self.quality_multipliers = {
            0: 1.0,    # High quality - standard rate
            1: 0.8,    # Medium quality - 20% discount
            2: 0.6     # Low quality - 40% discount
        }
        self.results = {
            'ensemble': {},
            'transformer': {},
            'lightgbm': {},
            'comparison': {},
            'tariff_analysis': {}
        }

        # Set memory-efficient configurations
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(2)  # Limit CPU threads
        
        self.setup_logging()
        self.create_output_directory()
        
        # Initialize analyzers with reduced settings
        self.ensemble_analyzer = AdvancedPowerQualityAnalyzer()
        self.transformer_analyzer = PowerQualityAnalysis()
        
        # Initialize results dictionary
        self.results = {
            'ensemble': {},
            'transformer': {},
            'lightgbm': {},
            'comparison': {},
            'tariff_analysis': {}
        }
        

    def generate_tariff_plots(self):
        """Generate comprehensive tariff analysis plots"""
        try:
            # 1. Tariff Distribution by Model and Quality Class
            plt.figure(figsize=(12, 6))
            
            models = ['transformer', 'ensemble', 'lightgbm']
            quality_labels = ['High', 'Medium', 'Low']
            
            data = []
            for model in models:
                predictions = self.results[model]['predictions']
                tariffs = self.calculate_tariffs(predictions)[0]
                true_labels = self.results[model]['true_labels']
                
                for i, pred in enumerate(predictions):
                    data.append({
                        'Model': model.capitalize(),
                        'Quality': quality_labels[true_labels[i]],
                        'Tariff': tariffs[i]
                    })
            
            df = pd.DataFrame(data)
            sns.boxplot(x='Quality', y='Tariff', hue='Model', data=df)
            plt.title('Tariff Distribution by Power Quality Class and Model')
            plt.savefig(self.output_dir / 'plots' / 'tariff_quality_distribution.png')
            plt.close()
            
            # 2. Revenue Impact Analysis
            plt.figure(figsize=(10, 6))
            revenue_data = []
            for model in models:
                tariffs = self.calculate_tariffs(self.results[model]['predictions'])[0]
                revenue_data.append({
                    'Model': model.capitalize(),
                    'Total Revenue': np.sum(tariffs),
                    'Mean Tariff': np.mean(tariffs),
                    'Std Tariff': np.std(tariffs)
                })
            
            revenue_df = pd.DataFrame(revenue_data)
            ax = revenue_df.plot(x='Model', y='Total Revenue', kind='bar')
            plt.title('Total Revenue Comparison by Model')
            plt.ylabel('Revenue Units')
            for i, v in enumerate(revenue_df['Total Revenue']):
                ax.text(i, v, f'${v:,.2f}', ha='center')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'revenue_comparison.png')
            plt.close()
            
            # 3. Tariff Accuracy Analysis
            plt.figure(figsize=(12, 6))
            accuracy_data = []
            for model in models:
                predictions = self.results[model]['predictions']
                true_labels = self.results[model]['true_labels']
                tariffs = self.calculate_tariffs(predictions)[0]
                true_tariffs = self.calculate_tariffs(true_labels)[0]
                
                mae = np.mean(np.abs(tariffs - true_tariffs))
                mape = np.mean(np.abs((tariffs - true_tariffs) / true_tariffs)) * 100
                
                accuracy_data.append({
                    'Model': model.capitalize(),
                    'MAE': mae,
                    'MAPE': mape
                })
            
            acc_df = pd.DataFrame(accuracy_data)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            acc_df.plot(x='Model', y='MAE', kind='bar', ax=ax1)
            ax1.set_title('Mean Absolute Error in Tariff Prediction')
            ax1.set_ylabel('MAE ($)')
            
            acc_df.plot(x='Model', y='MAPE', kind='bar', ax=ax2)
            ax2.set_title('Mean Absolute Percentage Error')
            ax2.set_ylabel('MAPE (%)')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'tariff_accuracy.png')
            plt.close()
            
            # 4. Economic Impact Analysis
            plt.figure(figsize=(10, 6))
            impact_data = []
            for model in models:
                predictions = self.results[model]['predictions']
                true_labels = self.results[model]['true_labels']
                
                # Calculate revenue difference
                pred_tariffs = self.calculate_tariffs(predictions)[0]
                true_tariffs = self.calculate_tariffs(true_labels)[0]
                revenue_diff = np.sum(pred_tariffs - true_tariffs)
                
                impact_data.append({
                    'Model': model.capitalize(),
                    'Revenue Impact': revenue_diff
                })
            
            impact_df = pd.DataFrame(impact_data)
            ax = impact_df.plot(x='Model', y='Revenue Impact', kind='bar')
            plt.title('Economic Impact of Model Predictions')
            plt.ylabel('Revenue Difference ($)')
            for i, v in enumerate(impact_df['Revenue Impact']):
                ax.text(i, v, f'${v:,.2f}', ha='center')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'economic_impact.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error generating tariff plots: {str(e)}")
            raise

    def calculate_tariffs(self, predictions):
        """Calculate tariffs based on power quality predictions"""
        try:
            tariffs = []
            for pred in predictions:
                tariff = self.base_tariff * self.quality_multipliers[pred]
                tariffs.append(tariff)
            
            tariffs = np.array(tariffs)
            
            # Calculate summary statistics
            tariff_stats = {
                'mean_tariff': np.mean(tariffs),
                'std_tariff': np.std(tariffs),
                'min_tariff': np.min(tariffs),
                'max_tariff': np.max(tariffs),
                'total_revenue': np.sum(tariffs)
            }
            
            return tariffs, tariff_stats
            
        except Exception as e:
            logging.error(f"Error calculating tariffs: {str(e)}")
            raise

    def perform_advanced_analysis(self, training_results):
        """Perform comprehensive analysis for publication"""
        try:
            analysis_results = {}
            
            # 1. Feature Importance Analysis (for ensemble)
            if hasattr(training_results['ensemble']['model'].named_steps['classifier'], 'feature_importances_'):
                feature_importance = training_results['ensemble']['model'].named_steps['classifier'].feature_importances_
                analysis_results['feature_importance'] = {
                    'importance_values': feature_importance,
                    'mean_importance': np.mean(feature_importance),
                    'std_importance': np.std(feature_importance)
                }
            
            # 2. Model Complexity Analysis
            complexity_metrics = {
                'transformer': {
                    'parameters': sum(p.numel() for p in training_results['transformer']['model'].parameters()),
                    'training_time': training_results['transformer']['training_time'],
                    'architecture': str(training_results['transformer']['model'])
                },
                'ensemble': {
                    'n_estimators': training_results['ensemble']['model'].named_steps['classifier'].n_estimators,
                    'training_time': training_results['ensemble']['training_time'],
                    'feature_count': training_results['ensemble']['model'].named_steps['classifier'].n_features_in_
                }
            }
            analysis_results['complexity_metrics'] = complexity_metrics
            
            # 3. Statistical Analysis
            transformer_preds = self.results['transformer']['predictions']
            ensemble_preds = self.results['ensemble']['predictions']
            true_labels = self.results['transformer']['true_labels']
            
            # McNemar's test
            contingency_table = np.zeros((2, 2))
            for i in range(len(true_labels)):
                trans_correct = transformer_preds[i] == true_labels[i]
                ens_correct = ensemble_preds[i] == true_labels[i]
                contingency_table[int(trans_correct), int(ens_correct)] += 1
            
            try:
                mcnemar_stat, p_value = stats.mcnemar(contingency_table, exact=True)
                analysis_results['statistical_tests'] = {
                    'mcnemar_statistic': mcnemar_stat,
                    'p_value': p_value
                }
            except Exception as e:
                logging.warning(f"McNemar's test failed: {str(e)}")
            
            # 4. Quality-wise Performance Analysis
            quality_metrics = {}
            for quality in range(3):  # 0: High, 1: Medium, 2: Low
                mask = true_labels == quality
                quality_metrics[f'quality_{quality}'] = {
                    'transformer_accuracy': accuracy_score(
                        true_labels[mask], transformer_preds[mask]
                    ),
                    'ensemble_accuracy': accuracy_score(
                        true_labels[mask], ensemble_preds[mask]
                    ),
                    'sample_count': np.sum(mask)
                }
            analysis_results['quality_metrics'] = quality_metrics
            
            # 5. Calculate and analyze tariffs
            transformer_tariffs, transformer_stats = self.calculate_tariffs(transformer_preds)
            ensemble_tariffs, ensemble_stats = self.calculate_tariffs(ensemble_preds)
            
            analysis_results['tariff_analysis'] = {
                'transformer': transformer_stats,
                'ensemble': ensemble_stats,
                'difference_stats': {
                    'mean_diff': np.mean(transformer_tariffs - ensemble_tariffs),
                    'std_diff': np.std(transformer_tariffs - ensemble_tariffs)
                }
            }
            
            return analysis_results
            
        except Exception as e:
            logging.error(f"Error in advanced analysis: {str(e)}")
            raise

    def generate_publication_plots(self):
        """Generate comprehensive plots for publication"""
        try:
            logging.info("Generating publication-quality plots...")
            
            # 1. Quality Distribution Plot
            plt.figure(figsize=(10, 6))
            quality_counts = pd.Series(self.results['transformer']['true_labels']).value_counts()
            sns.barplot(x=quality_counts.index, y=quality_counts.values)
            plt.title('Distribution of Power Quality Classes')
            plt.xlabel('Quality Class')
            plt.ylabel('Count')
            plt.savefig(self.output_dir / 'plots' / 'quality_distribution.png')
            plt.close()
            
            # 2. Feature Importance Plot
            if 'feature_importance' in self.results:
                plt.figure(figsize=(12, 6))
                importance_df = pd.DataFrame(self.results['feature_importance'])
                sns.barplot(data=importance_df, x='Feature', y='Importance')
                plt.xticks(rotation=45)
                plt.title('Feature Importance Analysis')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'plots' / 'feature_importance.png')
                plt.close()
            
            # 3. Tariff Distribution Plot
            plt.figure(figsize=(10, 6))
            transformer_tariffs = self.results['tariff_analysis']['transformer']['tariffs']
            ensemble_tariffs = self.results['tariff_analysis']['ensemble']['tariffs']
            
            plt.hist(transformer_tariffs, alpha=0.5, label='Transformer', bins=20)
            plt.hist(ensemble_tariffs, alpha=0.5, label='Ensemble', bins=20)
            plt.title('Tariff Distribution by Model')
            plt.xlabel('Tariff Rate')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(self.output_dir / 'plots' / 'tariff_distribution.png')
            plt.close()
            
            # 4. Quality-wise Performance Comparison
            plt.figure(figsize=(12, 6))
            quality_metrics = self.results['quality_metrics']
            
            qualities = list(quality_metrics.keys())
            transformer_acc = [metrics['transformer_accuracy'] for metrics in quality_metrics.values()]
            ensemble_acc = [metrics['ensemble_accuracy'] for metrics in quality_metrics.values()]
            
            x = np.arange(len(qualities))
            width = 0.35
            
            plt.bar(x - width/2, transformer_acc, width, label='Transformer')
            plt.bar(x + width/2, ensemble_acc, width, label='Ensemble')
            plt.xlabel('Quality Class')
            plt.ylabel('Accuracy')
            plt.title('Quality-wise Performance Comparison')
            plt.xticks(x, qualities)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'quality_performance.png')
            plt.close()
            
            # 5. Model Training Time Comparison
            plt.figure(figsize=(8, 6))
            times = [
                self.results['complexity_metrics']['transformer']['training_time'],
                self.results['complexity_metrics']['ensemble']['training_time']
            ]
            plt.bar(['Transformer', 'Ensemble'], times)
            plt.title('Model Training Time Comparison')
            plt.ylabel('Time (seconds)')
            plt.savefig(self.output_dir / 'plots' / 'training_time.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error generating publication plots: {str(e)}")
            raise

    # def evaluate_models(self, training_results):
    #     """Enhanced evaluation with publication metrics"""
    #     try:
    #         # Existing evaluation code...
    #         transformer_metrics = self._evaluate_transformer(
    #             training_results['transformer']['model'],
    #             training_results['transformer']['data_loader'],
    #             training_results['transformer']['test_data'][1]
    #         )
            
    #         ensemble_metrics = self._evaluate_ensemble(
    #             training_results['ensemble']['model'],
    #             *training_results['ensemble']['test_data']
    #         )
            
    #         # Store basic results
    #         self.results['transformer'].update(transformer_metrics)
    #         self.results['ensemble'].update(ensemble_metrics)
            
    #         # Perform advanced analysis
    #         advanced_analysis = self.perform_advanced_analysis(training_results)
    #         self.results.update(advanced_analysis)
            
    #         # Calculate tariffs
    #         transformer_tariffs, transformer_stats = self.calculate_tariffs(
    #             self.results['transformer']['predictions']
    #         )
    #         ensemble_tariffs, ensemble_stats = self.calculate_tariffs(
    #             self.results['ensemble']['predictions']
    #         )
            
    #         self.results['tariff_analysis'] = {
    #             'transformer': {
    #                 'tariffs': transformer_tariffs,
    #                 **transformer_stats
    #             },
    #             'ensemble': {
    #                 'tariffs': ensemble_tariffs,
    #                 **ensemble_stats
    #             }
    #         }
            
    #         return self.results
            
    #     except Exception as e:
    #         logging.error(f"Error in enhanced evaluation: {str(e)}")
    #         raise
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
    
    def create_output_directory(self):
        """Create directory structure for outputs"""
        self.output_dir = Path('comparison_results')
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
    
    def generate_dataset(self, n_samples=100):
        """Generate synthetic dataset with memory management"""
        logging.info(f"Generating {n_samples} synthetic signals...")
        try:
            # Generate data in smaller batches
            batch_size = 5
            signals = []
            labels = []
            
            for i in range(0, n_samples, batch_size):
                current_batch = min(batch_size, n_samples - i)
                batch_signals, batch_labels = self.transformer_analyzer.generate_synthetic_dataset(
                    n_samples=current_batch
                )
                
                signals.append(batch_signals)
                labels.append(batch_labels)
                
                # Force garbage collection after each batch
                gc.collect()
                
                # Log progress
                logging.info(f"Generated batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}")
            
            # Combine batches
            signals = np.concatenate(signals)
            labels = np.concatenate(labels)
            
            return signals, labels
            
        except Exception as e:
            logging.error(f"Error generating dataset: {str(e)}")
            raise
    
    def train_models(self, signals, labels):
        """Train both models with memory-efficient settings"""
        try:
            # Memory-efficient settings for data preparation
             # Memory-efficient settings for data preparation
            batch_size = 4  # Reduced batch size
            logging.info("Preparing data for transformer...")
            train_loader, test_loader, X_test_trans, y_test_trans = \
                self.transformer_analyzer.prepare_data(signals, labels, batch_size=batch_size)
            
            # Initialize results dictionary
            results = {}
            
            logging.info("Training LightGBM model...")
            start_time = time.time()
            features = self.lightgbm_analyzer.extract_enhanced_features(signals)
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            lightgbm_model = self.lightgbm_analyzer.build_model()
            lightgbm_model.fit(X_train, y_train)
            lightgbm_training_time = time.time() - start_time
            
            results['lightgbm'] = {
                'model': lightgbm_model,
                'training_time': lightgbm_training_time,
                'test_data': (X_test, y_test),
                'feature_names': self.lightgbm_analyzer.feature_names
            }
            
            # Train transformer with reduced parameters
            logging.info("Training transformer model...")
            start_time = time.time()
            transformer_model, transformer_history = \
                self.transformer_analyzer.train_transformer(
                    train_loader, 
                    test_loader,
                    epochs=5  # Reduced epochs
                )
            transformer_training_time = time.time() - start_time
            
            # Clear memory before ensemble training
            gc.collect()
            
            # Train ensemble with reduced complexity
            logging.info("Training ensemble model...")
            start_time = time.time()
            features = self.ensemble_analyzer.extract_enhanced_features(signals)
            X_train, X_test_ens, y_train, y_test_ens = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            ensemble_model = self.ensemble_analyzer.build_advanced_model()
            ensemble_model.fit(X_train, y_train)
            ensemble_training_time = time.time() - start_time
            
            results['transformer'] = {
                'model': transformer_model,
                'history': transformer_history,
                'training_time': transformer_training_time,
                'test_data': (X_test_trans, y_test_trans),
                'data_loader': test_loader
            }
            
            results['ensemble'] = {
                'model': ensemble_model,
                'training_time': ensemble_training_time,
                'test_data': (X_test_ens, y_test_ens)
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            raise
    
    def _evaluate_transformer(self, model, test_loader, y_test):
        """Evaluate transformer model with memory efficiency"""
        model.eval()
        predictions = []
        probabilities = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                signals, labels = batch
                signals = signals.to(self.transformer_analyzer.device)
                outputs = model(signals)
                probs = F.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

                del signals, outputs, probs, preds
                gc.collect()
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'metrics': classification_report(true_labels, predictions, output_dict=True)
        }
    
    def _evaluate_ensemble(self, model, X_test, y_test):
        """Evaluate ensemble model"""
        batch_size = 100
        predictions = []
        probabilities = []
        
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_pred = model.predict(batch_X)
            batch_prob = model.predict_proba(batch_X)
            
            predictions.extend(batch_pred)
            probabilities.extend(batch_prob)
            
            gc.collect()
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': y_test,
            'metrics': classification_report(y_test, predictions, output_dict=True)
        }
    
    def evaluate_models(self, training_results):
        """Enhanced model evaluation with comprehensive metrics for publication"""
        try:
        # 1. Basic Model Evaluation (existing functionality)
            logging.info("Evaluating transformer model...")
            transformer_metrics = self._evaluate_transformer(
            training_results['transformer']['model'],
            training_results['transformer']['data_loader'],
            training_results['transformer']['test_data'][1]
            )
            gc.collect()
        
            logging.info("Evaluating ensemble model...")
            ensemble_metrics = self._evaluate_ensemble(
            training_results['ensemble']['model'],
            *training_results['ensemble']['test_data']
            )
            gc.collect()

            logging.info("Evaluating LightGBM model...")
            lightgbm_metrics = self._evaluate_ensemble(  # We can reuse this method since LightGBM has the same interface
            training_results['lightgbm']['model'],
            *training_results['lightgbm']['test_data']
            )
            gc.collect()
        # Store basic results
            self.results['transformer'].update(transformer_metrics)
            self.results['ensemble'].update(ensemble_metrics)
            self.results['lightgbm'].update(lightgbm_metrics)

        
        # Compare training times
            self.results['comparison']['training_time_ratio'] = {
            'transformer_to_ensemble': training_results['transformer']['training_time'] / training_results['ensemble']['training_time'],
            'transformer_to_lightgbm': training_results['transformer']['training_time'] / training_results['lightgbm']['training_time'],
            'ensemble_to_lightgbm': training_results['ensemble']['training_time'] / training_results['lightgbm']['training_time']
             }
            # Calculate tariffs for all models
            for model_name in ['transformer', 'ensemble', 'lightgbm']:
             tariffs, tariff_stats = self.calculate_tariffs(self.results[model_name]['predictions'])
             self.results['tariff_analysis'][model_name] = {
                'tariffs': tariffs,
                **tariff_stats
            }
        # 2. Calculate Tariffs
            logging.info("Calculating tariffs...")
            transformer_tariffs, transformer_tariff_stats = self.calculate_tariffs(
            self.results['transformer']['predictions']
            )
            ensemble_tariffs, ensemble_tariff_stats = self.calculate_tariffs(
            self.results['ensemble']['predictions']
            )
        
            self.results['tariff_analysis'] = {
            'transformer': {
                'tariffs': transformer_tariffs,
                **transformer_tariff_stats
            },
            'ensemble': {
                'tariffs': ensemble_tariffs,
                **ensemble_tariff_stats
            }
            }

        # 3. Statistical Analysis
            logging.info("Performing statistical analysis...")
            transformer_preds = self.results['transformer']['predictions']
            ensemble_preds = self.results['ensemble']['predictions']
            true_labels = self.results['transformer']['true_labels']

        # McNemar's test for model comparison
            contingency_table = np.zeros((2, 2))
            for i in range(len(true_labels)):
                trans_correct = transformer_preds[i] == true_labels[i]
                ens_correct = ensemble_preds[i] == true_labels[i]
                contingency_table[int(trans_correct), int(ens_correct)] += 1
        
            try:
             mcnemar_stat, p_value = stats.mcnemar(contingency_table, exact=True)
             self.results['statistical_tests'] = {
                'mcnemar_statistic': mcnemar_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
            except Exception as e:
             logging.warning(f"McNemar's test failed: {str(e)}")

        # 4. Quality-wise Analysis
            logging.info("Performing quality-wise analysis...")
            quality_metrics = {}
            for quality in range(3):  # 0: High, 1: Medium, 2: Low
                mask = true_labels == quality
                quality_name = f'quality_{quality}'
            
            # Calculate metrics for each quality level
                quality_metrics[quality_name] = {
                'transformer_accuracy': accuracy_score(
                    true_labels[mask], transformer_preds[mask]
                ),
                'ensemble_accuracy': accuracy_score(
                    true_labels[mask], ensemble_preds[mask]
                ),
                'sample_count': np.sum(mask),
                'transformer_predictions': {
                    'precision': precision_score(true_labels[mask], transformer_preds[mask], average='weighted'),
                    'recall': recall_score(true_labels[mask], transformer_preds[mask], average='weighted'),
                    'f1': f1_score(true_labels[mask], transformer_preds[mask], average='weighted')
                },
                'ensemble_predictions': {
                    'precision': precision_score(true_labels[mask], ensemble_preds[mask], average='weighted'),
                    'recall': recall_score(true_labels[mask], ensemble_preds[mask], average='weighted'),
                    'f1': f1_score(true_labels[mask], ensemble_preds[mask], average='weighted')
                }
            }
        
            self.results['quality_metrics'] = quality_metrics

        # 5. Model Complexity Analysis
            logging.info("Analyzing model complexity...")
            self.results['complexity_metrics'] = {
            'transformer': {
                'parameters': sum(p.numel() for p in training_results['transformer']['model'].parameters()),
                'training_time': training_results['transformer']['training_time'],
                'architecture': str(training_results['transformer']['model']),
                'learning_rate': training_results['transformer'].get('learning_rate', 'N/A'),
                'batch_size': training_results['transformer'].get('batch_size', 'N/A')
            },
            'ensemble': {
                'n_estimators': training_results['ensemble']['model'].named_steps['classifier'].n_estimators,
                'training_time': training_results['ensemble']['training_time'],
                'feature_count': training_results['ensemble']['model'].named_steps['classifier'].n_features_in_,
                'max_depth': training_results['ensemble']['model'].named_steps['classifier'].max_depth
            },
            'lightgbm': {
            'n_estimators': training_results['lightgbm']['model'].named_steps['classifier'].n_estimators,
            'training_time': training_results['lightgbm']['training_time'],
            'feature_count': len(training_results['lightgbm']['feature_names']),
            'max_depth': training_results['lightgbm']['model'].named_steps['classifier'].get_params()['max_depth']
            }
        }

        # 6. Performance Stability Analysis
            logging.info("Analyzing performance stability...")
            self.results['stability_metrics'] = {
            'transformer': {
                'prediction_variance': np.var(self.results['transformer']['probabilities'], axis=0),
                'confidence_mean': np.mean(np.max(self.results['transformer']['probabilities'], axis=1)),
                'confidence_std': np.std(np.max(self.results['transformer']['probabilities'], axis=1))
            },
            'ensemble': {
                'prediction_variance': np.var(self.results['ensemble']['probabilities'], axis=0),
                'confidence_mean': np.mean(np.max(self.results['ensemble']['probabilities'], axis=1)),
                'confidence_std': np.std(np.max(self.results['ensemble']['probabilities'], axis=1))
            }
        }

        # 7. Calculate Combined Performance Score
            logging.info("Calculating combined performance scores...")
            self.results['performance_scores'] = {
            'transformer': {
                'accuracy_score': self.results['transformer']['metrics']['accuracy'],
                'stability_score': 1 - self.results['stability_metrics']['transformer']['confidence_std'],
                'speed_score': 1 / (1 + training_results['transformer']['training_time'])
            },
            'ensemble': {
                'accuracy_score': self.results['ensemble']['metrics']['accuracy'],
                'stability_score': 1 - self.results['stability_metrics']['ensemble']['confidence_std'],
                'speed_score': 1 / (1 + training_results['ensemble']['training_time'])
            }
        }

        # Calculate weighted combined score
            for model in ['transformer', 'ensemble']:
             scores = self.results['performance_scores'][model]
             self.results['performance_scores'][model]['combined_score'] = (
                0.5 * scores['accuracy_score'] +
                0.3 * scores['stability_score'] +
                0.2 * scores['speed_score']
             )

             logging.info("Model evaluation completed successfully")
            return self.results
        
        except Exception as e:
         logging.error(f"Error evaluating models: {str(e)}")
         logging.debug(f"Detailed error information:", exc_info=True)
         raise

    def generate_comparative_plots(self):
        """Generate plots with memory management"""
        logging.info("Generating comparative plots...")
        try:
            # Create figure for accuracy comparison
            plt.figure(figsize=(10, 6))
            accuracies = {
                'Transformer': self.results['transformer']['metrics']['accuracy'],
                'Ensemble': self.results['ensemble']['metrics']['accuracy'],
                'LightGBM': self.results['lightgbm']['metrics']['accuracy']

            }
            plt.bar(list(accuracies.keys()), list(accuracies.values()))
            plt.title('Model Accuracy Comparison')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            for i, v in enumerate(accuracies.values()):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
            plt.savefig(self.output_dir / 'plots' / 'accuracy_comparison.png')
            plt.close()

           # Create confusion matrices - Fixed version with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))  # Changed to create three axes
            labels = ['High', 'Medium', 'Low']
            
            # Confusion matrices for all three models
            models = {
                'Transformer': (self.results['transformer'], ax1),
                'Ensemble': (self.results['ensemble'], ax2),
                'LightGBM': (self.results['lightgbm'], ax3)
             }
        
            for model_name, (results, ax) in models.items():
                cm = confusion_matrix(
                     results['true_labels'],
                     results['predictions']
            )
                sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                            xticklabels=labels, yticklabels=labels, cmap='Blues')
                ax.set_title(f'{model_name} Confusion Matrix')
        
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'confusion_matrices.png')
            plt.close()

            # Create ROC curves
            plt.figure(figsize=(10, 8))
            for i, class_name in enumerate(['High', 'Medium', 'Low']):
                # Transformer ROC
                fpr, tpr, _ = roc_curve(
                    self.results['transformer']['true_labels'] == i,
                    self.results['transformer']['probabilities'][:, i]
                )
                transformer_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Transformer - {class_name} (AUC = {transformer_auc:.2f})')
                
                # Ensemble ROC
                fpr, tpr, _ = roc_curve(
                    self.results['ensemble']['true_labels'] == i,
                    self.results['ensemble']['probabilities'][:, i]
                )
                ensemble_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, '--', label=f'Ensemble - {class_name} (AUC = {ensemble_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'roc_curves.png')
            plt.close()

            # Create prediction confidence distribution
            plt.figure(figsize=(10, 6))
            transformer_conf = np.max(self.results['transformer']['probabilities'], axis=1)
            ensemble_conf = np.max(self.results['ensemble']['probabilities'], axis=1)
            
            plt.hist(transformer_conf, alpha=0.5, label='Transformer', bins=20)
            plt.hist(ensemble_conf, alpha=0.5, label='Ensemble', bins=20)
            plt.xlabel('Prediction Confidence')
            plt.ylabel('Count')
            plt.title('Prediction Confidence Distribution')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'prediction_confidence.png')
            plt.close()

            # Create training history plot
            if 'history' in self.results['transformer']:
                history = self.results['transformer']['history']
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.plot(history['train_losses'], label='Train')
                ax1.plot(history['test_losses'], label='Validation')
                ax1.set_title('Training Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                
                ax2.plot(history['train_accuracies'], label='Train')
                ax2.plot(history['test_accuracies'], label='Validation')
                ax2.set_title('Training Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'plots' / 'training_history.png')
                plt.close()

        except Exception as e:
            logging.error(f"Error generating plots: {str(e)}")
            raise
        finally:
            plt.close('all')
            gc.collect()
    
    # In comparison.py, modify the save_results method

    def save_results(self):
        """Save results with enhanced publication metrics"""
        try:
        # Save metrics as JSON
            metrics_path = self.output_dir / 'metrics' / 'results.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.results, f, indent=4, default=str)
        
        # Generate basic summary report (existing functionality)
            report_path = self.output_dir / 'metrics' / 'summary_report.txt'
            with open(report_path, 'w') as f:
                f.write("Power Quality Classification - Model Comparison Report\n")
                f.write("================================================\n\n")
            
            # Model accuracies
                f.write("1. Model Accuracies:\n")
                for model in ['transformer', 'ensemble', 'lightgbm']:
                    if model in self.results and 'metrics' in self.results[model]:
                        f.write(f"   {model.capitalize()}: {self.results[model]['metrics']['accuracy']:.4f}\n")
                f.write("\n")
                
                # Training time comparison
                f.write("2. Training Time Comparison:\n")
                if 'comparison' in self.results and 'training_time_ratio' in self.results['comparison']:
                    ratios = self.results['comparison']['training_time_ratio']
                    if isinstance(ratios, dict):
                        for ratio_name, ratio_value in ratios.items():
                            f.write(f"   {ratio_name}: {ratio_value:.2f}\n")
                    else:
                        f.write(f"   Ratio (Transformer/Ensemble): {ratios:.2f}\n")
                f.write("\n")
                
                # Class-wise performance
                f.write("3. Class-wise Performance:\n")
                for model in ['transformer', 'ensemble', 'lightgbm']:
                    if model in self.results and 'metrics' in self.results[model]:
                        f.write(f"\n   {model.capitalize()} Model:\n")
                        for class_name in ['0', '1', '2']:
                            if class_name in self.results[model]['metrics']:
                                metrics = self.results[model]['metrics'][class_name]
                                f.write(f"   Class {class_name} (")
                                if class_name == '0':
                                    f.write("High Quality):\n")
                                elif class_name == '1':
                                    f.write("Medium Quality):\n")
                                else:
                                    f.write("Low Quality):\n")
                                f.write(f"      Precision: {metrics['precision']:.4f}\n")
                                f.write(f"      Recall: {metrics['recall']:.4f}\n")
                                f.write(f"      F1-score: {metrics['f1-score']:.4f}\n")
                
                # Tariff Analysis
                f.write("\n4. Tariff Analysis:\n")
                if 'tariff_analysis' in self.results:
                    for model in ['transformer', 'ensemble', 'lightgbm']:
                        if model in self.results['tariff_analysis']:
                            stats = self.results['tariff_analysis'][model]
                            f.write(f"\n   {model.capitalize()} Model:\n")
                            if isinstance(stats, dict):  # Check if stats is a dictionary
                                for stat_name in ['mean_tariff', 'min_tariff', 'max_tariff', 'total_revenue']:
                                    if stat_name in stats:
                                        formatted_name = stat_name.replace('_', ' ').title()
                                        if 'revenue' in stat_name:
                                            f.write(f"      {formatted_name}: ${stats[stat_name]:.2f}\n")
                                        else:
                                            f.write(f"      {formatted_name}: ${stats[stat_name]:.4f}\n")

                # Statistical Analysis
                f.write("\n5. Statistical Analysis:\n")
                if 'statistical_tests' in self.results:
                    stats = self.results['statistical_tests']
                    f.write(f"   McNemar's Test Statistic: {stats.get('mcnemar_statistic', 'N/A'):.4f}\n")
                    f.write(f"   P-value: {stats.get('p_value', 'N/A'):.4f}\n")
                    f.write(f"   Statistical Significance: {'Yes' if stats.get('p_value', 1) < 0.05 else 'No'}\n")

                # Quality-wise Analysis
                f.write("\n6. Quality-wise Analysis:\n")
                if 'quality_metrics' in self.results:
                    for quality, metrics in self.results['quality_metrics'].items():
                        quality_name = 'High' if quality == 'quality_0' else 'Medium' if quality == 'quality_1' else 'Low'
                        f.write(f"\n   {quality_name} Quality Power:\n")
                        for model in ['transformer', 'ensemble', 'lightgbm']:
                            if f'{model}_accuracy' in metrics:
                                f.write(f"      {model.capitalize()} Accuracy: {metrics[f'{model}_accuracy']:.4f}\n")
                        f.write(f"      Sample Count: {metrics.get('sample_count', 'N/A')}\n")

            # Save detailed CSV reports
            metrics_df = pd.DataFrame({
                'Metric': self.results['comparison'].keys(),
                'Value': self.results['comparison'].values()
            })
            metrics_df.to_csv(self.output_dir / 'metrics' / 'detailed_metrics.csv', index=False)
            
            logging.info("Results saved successfully")
        
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            logging.debug(f"Detailed error: {str(e)}", exc_info=True)
            raise

def main():
    # Track memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    try:
        # Set memory-efficient configurations
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(2)  # Limit CPU threads
        
        # Initialize comparison framework
        logging.info("Initializing comparison framework...")
        comparison = PowerQualityComparison()
        
        # Generate smaller dataset with progress tracking
        logging.info("Generating dataset...")
        signals, labels = comparison.generate_dataset(n_samples=10)  # Start with small sample
        gc.collect()
        
        # Train models with memory management
        logging.info("Training models...")
        training_results = comparison.train_models(signals, labels)
        gc.collect()
        
        # Evaluate models
        logging.info("Evaluating models...")
        results = comparison.evaluate_models(training_results)
        del training_results  # Free memory
        gc.collect()
        
        # Generate plots with memory cleanup
        logging.info("Generating plots...")
        comparison.generate_comparative_plots()
        gc.collect()
        
        # Save results
        logging.info("Saving results...")
        comparison.save_results()
        
        # Log memory usage and completion
        final_memory = process.memory_info().rss / 1024 / 1024
        logging.info(f"Memory usage: {final_memory - initial_memory:.2f} MB")
        logging.info("Comparison completed successfully")
        
    except Exception as e:
        logging.error(f"Comparison failed: {str(e)}")
        raise
    finally:
        # Final cleanup
        gc.collect()
        plt.close('all')

if __name__ == "__main__":
    # Set up basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
                        
                        