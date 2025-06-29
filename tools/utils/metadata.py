"""Metadata extraction and management utilities."""

import os
import json
import pickle
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

from ..config import OUTPUT_DIR, logger


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract metadata from model filename patterns.
    
    Expected pattern: train_{model_type}_price_predictor_{symbol}_model_{timestamp}.pkl
    
    Args:
        filename: Model filename
        
    Returns:
        Dictionary with extracted metadata
    """
    try:
        # Remove .pkl extension
        base_name = filename.replace('.pkl', '')
        parts = base_name.split('_')
        
        metadata = {
            'model_type': 'unknown',
            'symbol': 'unknown',
            'target_days': 1,
            'timestamp': 'unknown',
            'features_count': 0,
            'training_type': 'unknown'
        }
        
        # Extract model type (usually parts[1])
        if len(parts) > 1:
            metadata['model_type'] = parts[1]
        
        # Extract symbol (usually before 'model')
        if 'model' in parts:
            model_index = parts.index('model')
            if model_index > 0:
                metadata['symbol'] = parts[model_index - 1]
        
        # Extract timestamp (usually last part)
        if len(parts) > 0:
            metadata['timestamp'] = parts[-1]
        
        # Extract training type (usually parts[0])
        if len(parts) > 0:
            metadata['training_type'] = parts[0]
        
        logger.debug(f"Extracted metadata from {filename}: {metadata}")
        return metadata
        
    except Exception as e:
        logger.warning(f"Warning: Could not extract metadata from filename {filename}: {str(e)}")
        return {
            'model_type': 'unknown',
            'symbol': 'unknown', 
            'target_days': 1,
            'timestamp': 'unknown',
            'features_count': 0,
            'training_type': 'unknown'
        }


def extract_symbol_from_filename(filename: str) -> str:
    """
    Extract stock symbol from filename.
    
    Args:
        filename: Any filename containing a stock symbol
        
    Returns:
        Extracted stock symbol or 'unknown'
    """
    try:
        # Common stock symbol patterns
        patterns = [
            r'[_\-]([A-Z]{1,5})[_\-]',  # Symbol between underscores/hyphens
            r'^([A-Z]{1,5})[_\-]',      # Symbol at start
            r'[_\-]([A-Z]{1,5})\.csv',  # Symbol before .csv
            r'[_\-]([A-Z]{1,5})\.pkl',  # Symbol before .pkl
            r'[_\-]([A-Z]{1,5})\.json', # Symbol before .json
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.upper())
            if match:
                symbol = match.group(1)
                # Validate symbol (basic check)
                if 1 <= len(symbol) <= 5 and symbol.isalpha():
                    return symbol
        
        # Fallback: look for any 1-5 letter uppercase sequence
        uppercase_sequences = re.findall(r'[A-Z]{1,5}', filename.upper())
        for seq in uppercase_sequences:
            if 1 <= len(seq) <= 5:
                return seq
        
        return 'unknown'
        
    except Exception as e:
        logger.warning(f"Warning: Could not extract symbol from {filename}: {str(e)}")
        return 'unknown'


def extract_timestamp_from_filename(filename: str) -> str:
    """
    Extract timestamp from filename.
    
    Args:
        filename: Filename containing timestamp
        
    Returns:
        Extracted timestamp or 'unknown'
    """
    try:
        # Common timestamp patterns
        patterns = [
            r'(\d{8}_\d{6})',    # YYYYMMDD_HHMMSS
            r'(\d{14})',         # YYYYMMDDHHMMSS
            r'(\d{8})',          # YYYYMMDD
            r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',  # YYYY-MM-DD_HH-MM-SS
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return 'unknown'
        
    except Exception as e:
        logger.warning(f"Warning: Could not extract timestamp from {filename}: {str(e)}")
        return 'unknown'


def generate_model_signature(model_type: str, symbol: str, params: Dict[str, Any]) -> str:
    """
    Generate a unique signature for a model based on its type and key parameters.
    
    Args:
        model_type: Type of model (e.g., 'xgboost', 'random_forest')
        symbol: Stock symbol
        params: Model parameters
        
    Returns:
        Human-readable model signature
    """
    try:
        key_params = get_key_parameters(model_type, params)
        
        if not key_params:
            return f"{model_type}_{symbol}_default"
        
        # Create signature from key parameters
        param_str = "_".join([f"{k}{v}" for k, v in key_params.items()])
        signature = f"{model_type}_{symbol}_{param_str}"
        
        # Clean signature (remove spaces, special chars)
        signature = re.sub(r'[^a-zA-Z0-9_]', '', signature)
        
        logger.debug(f"Generated signature: {signature}")
        return signature
        
    except Exception as e:
        logger.warning(f"Warning: Could not generate signature: {str(e)}")
        return f"{model_type}_{symbol}_error"


def get_key_parameters(model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the most important parameters for model identification.
    
    Args:
        model_type: Type of model
        params: Full parameter dictionary
        
    Returns:
        Dictionary with key parameters only
    """
    try:
        # Define key parameters for each model type
        key_param_map = {
            'xgboost': ['n_estimators', 'max_depth', 'learning_rate'],
            'random_forest': ['n_estimators', 'max_depth'],
            'svr': ['C', 'kernel'],
            'gradient_boosting': ['n_estimators', 'learning_rate', 'max_depth'],
            'ridge_regression': ['alpha'],
            'extra_trees': ['n_estimators', 'max_depth']
        }
        
        key_params = {}
        param_names = key_param_map.get(model_type, [])
        
        for param_name in param_names:
            if param_name in params:
                value = params[param_name]
                # Simplify value for signature
                if isinstance(value, float):
                    key_params[param_name] = f"{value:.2f}".rstrip('0').rstrip('.')
                else:
                    key_params[param_name] = str(value)
        
        return key_params
        
    except Exception as e:
        logger.warning(f"Warning: Could not extract key parameters: {str(e)}")
        return {}


def load_model_metadata(model_file: str) -> Dict[str, Any]:
    """
    Load comprehensive metadata from a saved model file.
    
    Args:
        model_file: Name of the model file
        
    Returns:
        Dictionary with model metadata
    """
    try:
        model_filepath = os.path.join(OUTPUT_DIR, model_file)
        
        # Try to load metadata from JSON file first
        json_file = model_file.replace('_model.pkl', '_results.json')
        json_filepath = os.path.join(OUTPUT_DIR, json_file)
        
        if os.path.exists(json_filepath):
            try:
                with open(json_filepath, 'r') as f:
                    metadata = json.load(f)
                logger.debug(f" load_model_metadata: Loaded metadata from JSON for {model_file}")
                return metadata
            except Exception as e:
                logger.warning(f"Warning: Could not load JSON metadata: {str(e)}")
        
        # Fallback: extract from pickle file
        if os.path.exists(model_filepath):
            try:
                with open(model_filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                metadata = {
                    'model_type': model_data.get('model_type', 'unknown'),
                    'symbol': model_data.get('symbol', 'unknown'),
                    'target_days': model_data.get('target_days', 1),
                    'feature_cols': model_data.get('feature_cols', []),
                    'features_count': len(model_data.get('feature_cols', [])),
                }
                
                # Try to extract additional metadata from filename
                filename_metadata = extract_metadata_from_filename(model_file)
                metadata.update(filename_metadata)
                
                logger.debug(f" load_model_metadata: Extracted metadata from pickle for {model_file}")
                return metadata
                
            except Exception as e:
                logger.warning(f"Warning: Could not load pickle metadata: {str(e)}")
        
        # Last resort: extract from filename only
        logger.warning(f"Warning: Using filename-only metadata for {model_file}")
        return extract_metadata_from_filename(model_file)
        
    except Exception as e:
        logger.warning(f"Warning: Could not load metadata for {model_file}: {str(e)}")
        return extract_metadata_from_filename(model_file)


def create_metadata_summary(metadata: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of metadata.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Formatted metadata summary string
    """
    try:
        summary_lines = []
        
        # Basic info
        summary_lines.append(f"Model Type: {metadata.get('model_type', 'Unknown')}")
        summary_lines.append(f"Symbol: {metadata.get('symbol', 'Unknown')}")
        summary_lines.append(f"Target Days: {metadata.get('target_days', 'Unknown')}")
        
        # Features
        features_count = metadata.get('features_count', 0)
        if features_count > 0:
            summary_lines.append(f"Features: {features_count}")
        
        # Performance metrics (if available)
        if 'performance' in metadata:
            perf = metadata['performance']
            if 'test_metrics' in perf:
                test = perf['test_metrics']
                if 'r2' in test:
                    summary_lines.append(f"R²: {test['r2']:.3f}")
                if 'rmse' in test:
                    summary_lines.append(f"RMSE: ${test['rmse']:.2f}")
        
        # Timestamp
        timestamp = metadata.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            try:
                # Try to parse and format timestamp
                if '_' in timestamp:
                    date_part, time_part = timestamp.split('_')
                    formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    summary_lines.append(f"Created: {formatted_date} {formatted_time}")
                else:
                    summary_lines.append(f"Created: {timestamp}")
            except:
                summary_lines.append(f"Created: {timestamp}")
        
        return " | ".join(summary_lines)
        
    except Exception as e:
        logger.warning(f"Warning: Could not create metadata summary: {str(e)}")
        return "Metadata summary unavailable"


def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean metadata dictionary.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Validated and cleaned metadata dictionary
    """
    try:
        validated = {}
        
        # Required fields with defaults
        validated['model_type'] = str(metadata.get('model_type', 'unknown'))
        validated['symbol'] = str(metadata.get('symbol', 'unknown')).upper()
        validated['target_days'] = int(metadata.get('target_days', 1))
        validated['features_count'] = int(metadata.get('features_count', 0))
        validated['timestamp'] = str(metadata.get('timestamp', 'unknown'))
        
        # Optional fields
        if 'feature_cols' in metadata:
            if isinstance(metadata['feature_cols'], list):
                validated['feature_cols'] = metadata['feature_cols']
        
        if 'performance' in metadata:
            if isinstance(metadata['performance'], dict):
                validated['performance'] = metadata['performance']
        
        if 'model_params' in metadata:
            if isinstance(metadata['model_params'], dict):
                validated['model_params'] = metadata['model_params']
        
        # Validation checks
        if validated['target_days'] < 1 or validated['target_days'] > 365:
            validated['target_days'] = 1
            logger.warning("Invalid target_days, reset to 1")
        
        if validated['features_count'] < 0:
            validated['features_count'] = 0
            logger.warning("Invalid features_count, reset to 0")
        
        # Validate symbol format
        symbol = validated['symbol']
        if not re.match(r'^[A-Z]{1,5}$', symbol) and symbol != 'UNKNOWN':
            validated['symbol'] = 'UNKNOWN'
            logger.warning(f"Invalid symbol format: {symbol}, reset to UNKNOWN")
        
        return validated
        
    except Exception as e:
        logger.error(f"Error validating metadata: {str(e)}")
        return {
            'model_type': 'unknown',
            'symbol': 'unknown',
            'target_days': 1,
            'features_count': 0,
            'timestamp': 'unknown'
        }


def find_related_files(base_filename: str) -> Dict[str, List[str]]:
    """
    Find files related to a base filename (e.g., model, results, predictions).
    
    Args:
        base_filename: Base filename to search for
        
    Returns:
        Dictionary with lists of related files by type
    """
    try:
        if not os.path.exists(OUTPUT_DIR):
            return {}
        
        # Extract base pattern from filename
        base_pattern = base_filename
        if '_model.pkl' in base_pattern:
            base_pattern = base_pattern.replace('_model.pkl', '')
        elif '_results.json' in base_pattern:
            base_pattern = base_pattern.replace('_results.json', '')
        elif '.csv' in base_pattern:
            base_pattern = base_pattern.replace('.csv', '')
        
        all_files = os.listdir(OUTPUT_DIR)
        related = {
            'models': [],
            'results': [],
            'predictions': [],
            'charts': [],
            'data': [],
            'other': []
        }
        
        for file in all_files:
            if base_pattern in file:
                if file.endswith('_model.pkl'):
                    related['models'].append(file)
                elif file.endswith('_results.json'):
                    related['results'].append(file)
                elif 'predictions' in file and file.endswith('.csv'):
                    related['predictions'].append(file)
                elif file.endswith('.html'):
                    related['charts'].append(file)
                elif file.endswith('.csv'):
                    related['data'].append(file)
                else:
                    related['other'].append(file)
        
        # Sort each category by modification time
        for category in related:
            related[category].sort(
                key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)),
                reverse=True
            )
        
        return related
        
    except Exception as e:
        logger.error(f"Error finding related files: {str(e)}")
        return {}