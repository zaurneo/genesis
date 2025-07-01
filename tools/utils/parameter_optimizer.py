"""Parameter optimization and validation utilities."""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal

# Import logging helpers
try:
    from ..logs.logging_helpers import log_info, log_success, log_warning, log_error, log_progress, safe_run
    _logging_helpers_available = True
except ImportError:
    _logging_helpers_available = False
    # Fallback to regular logger if logging_helpers not available
    def log_info(msg, **kwargs): logger.info(msg)
    def log_success(msg, **kwargs): logger.info(msg)
    def log_warning(msg, **kwargs): logger.warning(msg) 
    def log_error(msg, **kwargs): logger.error(msg)
    def log_progress(msg, **kwargs): logger.info(msg)
    def safe_run(func): return func

from ..config import PARAMETER_SCHEMAS, MODELING_CONTEXTS, VALIDATION_RULES, logger


@safe_run
def validate_model_parameters_impl(
    model_type: str,
    parameters: Dict[str, Any],
    symbol: Optional[str] = None,
    target_days: int = 1
) -> str:
    """
    Validate model parameters against schema and provide optimization suggestions.
    
    This function checks if provided parameters are valid for the specified model type,
    suggests improvements, and provides warnings about potential issues.
    
    Args:
        model_type: Type of model ('xgboost', 'random_forest', etc.)
        parameters: Dictionary of parameters to validate
        symbol: Stock symbol (for context-specific validation)
        target_days: Prediction horizon (for parameter optimization)
        
    Returns:
        String with validation results and recommendations
    """
    log_info(f"validate_model_parameters: Validating {model_type} parameters...")
    
    try:
        if model_type not in PARAMETER_SCHEMAS:
            available_types = ', '.join(PARAMETER_SCHEMAS.keys())
            result = f"validate_model_parameters: Unknown model type '{model_type}'. Available: {available_types}"
            log_error(f"validate_model_parameters: {result}")
            return result
        
        schema = PARAMETER_SCHEMAS[model_type]
        validation_results = []
        warnings = []
        suggestions = []
        
        # Check required parameters
        required_params = schema.get('required', [])
        missing_required = [param for param in required_params if param not in parameters]
        
        if missing_required:
            warnings.append(f"Missing required parameters: {', '.join(missing_required)}")
        
        # Validate each parameter
        for param_name, param_value in parameters.items():
            validation_result = validate_single_parameter(param_name, param_value, model_type)
            if validation_result:
                validation_results.append(validation_result)
        
        # Check parameter combinations
        combination_issues = check_parameter_combinations(parameters, model_type)
        if combination_issues:
            warnings.extend(combination_issues)
        
        # Generate optimization suggestions
        optimization_suggestions = suggest_parameter_optimizations(
            model_type, parameters, target_days
        )
        if optimization_suggestions:
            suggestions.extend(optimization_suggestions)
        
        # Context-specific recommendations
        context_recommendations = get_context_recommendations(model_type, target_days)
        if context_recommendations:
            suggestions.extend(context_recommendations)
        
        # Generate comprehensive report
        summary = f"""validate_model_parameters: Parameter validation completed for {model_type}:

 VALIDATION OVERVIEW:
- Model Type: {model_type.replace('_', ' ').title()}
- Parameters Checked: {len(parameters)}
- Validation Status: {'PASSED' if not warnings else 'WARNINGS'}

⚙️ PARAMETER ANALYSIS:
{chr(10).join([f"  ✓ {result}" for result in validation_results]) if validation_results else "  • No specific parameter issues found"}

 WARNINGS ({len(warnings)}):
{chr(10).join([f"  • {warning}" for warning in warnings]) if warnings else "  • No warnings"}

 OPTIMIZATION SUGGESTIONS ({len(suggestions)}):
{chr(10).join([f"  • {suggestion}" for suggestion in suggestions]) if suggestions else "  • Parameters appear well-optimized"}

 PARAMETER SUMMARY:
{format_parameter_summary(parameters, schema)}

 RECOMMENDATIONS:
- Overall Assessment: {'Parameters need attention' if warnings else 'Parameters look good'}
- Risk Level: {assess_parameter_risk(parameters, model_type)}
- Performance Expectation: {predict_performance_level(parameters, model_type)}
- Next Steps: {'Address warnings before training' if warnings else 'Ready for model training'}
"""
        
        log_info(f"validate_model_parameters: Validated {len(parameters)} parameters for {model_type}")
        return summary
        
    except Exception as e:
        error_msg = f"validate_model_parameters: Error validating parameters: {str(e)}"
        log_error(f"validate_model_parameters: {error_msg}")
        return error_msg


def get_model_selection_guide_impl(
    context: Literal["short_term_trading", "medium_term_investing", "long_term_investing", "high_volatility"] = "medium_term_investing",
    target_days: int = 1,
    available_features: int = 10,
    computational_budget: Literal["low", "medium", "high"] = "medium"
) -> str:
    """
    AI decision support for model selection based on context and constraints.
    
    This function provides intelligent recommendations for model type and parameters
    based on trading context, prediction horizon, and computational constraints.
    
    Args:
        context: Trading/investment context
        target_days: Number of days ahead to predict
        available_features: Number of technical indicators available
        computational_budget: Available computational resources
        
    Returns:
        String with model selection recommendations and reasoning
    """
    log_info(f" get_model_selection_guide: Generating recommendations for {context} context...")
    
    try:
        if context not in MODELING_CONTEXTS:
            available_contexts = ', '.join(MODELING_CONTEXTS.keys())
            result = f"get_model_selection_guide: Unknown context '{context}'. Available: {available_contexts}"
            log_error(f"get_model_selection_guide: {result}")
            return result
        
        context_info = MODELING_CONTEXTS[context]
        
        # Generate model rankings based on context
        model_rankings = rank_models_for_context(context, target_days, available_features, computational_budget)
        
        # Get parameter recommendations for top models
        param_recommendations = {}
        for model_type in model_rankings[:3]:  # Top 3 models
            param_recommendations[model_type] = get_recommended_parameters(
                model_type, context, target_days, computational_budget
            )
        
        # Generate feature recommendations
        feature_recommendations = get_feature_recommendations(context, available_features)
        
        # Assessment and warnings
        assessments = generate_context_assessments(context, target_days, available_features, computational_budget)
        
        # Generate comprehensive guide
        summary = f"""get_model_selection_guide: AI model selection recommendations:

 CONTEXT ANALYSIS:
- Trading Context: {context.replace('_', ' ').title()}
- Prediction Horizon: {target_days} day{'s' if target_days != 1 else ''}
- Available Features: {available_features}
- Computational Budget: {computational_budget.title()}
- Context Description: {context_info['description']}

 RECOMMENDED MODELS (Ranked):

1. 🥇 {model_rankings[0].replace('_', ' ').title()}
   - Rationale: {get_model_rationale(model_rankings[0], context)}
   - Parameters: {format_param_recommendations(param_recommendations.get(model_rankings[0], {}))}

2. 🥈 {model_rankings[1].replace('_', ' ').title()}
   - Rationale: {get_model_rationale(model_rankings[1], context)}
   - Parameters: {format_param_recommendations(param_recommendations.get(model_rankings[1], {}))}

3. 🥉 {model_rankings[2].replace('_', ' ').title()}
   - Rationale: {get_model_rationale(model_rankings[2], context)}
   - Parameters: {format_param_recommendations(param_recommendations.get(model_rankings[2], {}))}

🔧 FEATURE RECOMMENDATIONS:
{chr(10).join([f"  • {rec}" for rec in feature_recommendations])}

 CONTEXT ASSESSMENT:
{chr(10).join([f"  • {assessment}" for assessment in assessments])}

 IMPLEMENTATION STRATEGY:
- Start With: {model_rankings[0].replace('_', ' ').title()} (best fit for your context)
- Fallback Options: Try {model_rankings[1]} if performance is insufficient
- Ensemble Approach: Combine top 2-3 models for robust predictions
- Parameter Tuning: Focus on {get_key_tuning_parameters(model_rankings[0])}

 IMPORTANT CONSIDERATIONS:
- Data Quality: Ensure sufficient historical data ({get_min_data_requirement(target_days)} records minimum)
- Feature Engineering: Technical indicators should match your trading timeframe
- Validation: Use time-series cross-validation for reliable performance estimates
- Risk Management: Monitor model performance regularly in live trading

🚀 NEXT STEPS:
1. Implement {model_rankings[0]} with recommended parameters
2. Train on enhanced data with technical indicators
3. Validate performance using backtesting
4. Compare with benchmark (buy-and-hold)
5. Deploy with appropriate risk controls
"""
        
        log_success(f"get_model_selection_guide: Generated recommendations for {context} context")
        return summary
        
    except Exception as e:
        error_msg = f"get_model_selection_guide: Error generating guide: {str(e)}"
        log_error(f"get_model_selection_guide: {error_msg}")
        return error_msg


def validate_single_parameter(param_name: str, param_value: Any, model_type: str) -> Optional[str]:
    """Validate a single parameter value."""
    try:
        validation_rules = VALIDATION_RULES.get('model_parameters', {})
        
        if param_name not in validation_rules:
            return None
        
        rule = validation_rules[param_name]
        
        # Type validation
        expected_types = rule.get('type')
        if not isinstance(expected_types, list):
            expected_types = [expected_types]
        
        if not any(isinstance(param_value, t) for t in expected_types if t is not None):
            return f"{param_name}: Invalid type {type(param_value).__name__}, expected {expected_types}"
        
        # Range validation
        if 'min' in rule and param_value is not None:
            if param_value < rule['min']:
                return f"{param_name}: Value {param_value} below minimum {rule['min']}"
        
        if 'max' in rule and param_value is not None:
            if param_value > rule['max']:
                return f"{param_name}: Value {param_value} above maximum {rule['max']}"
        
        return f"{param_name}: {param_value} ✓"
        
    except Exception as e:
        return f"{param_name}: Validation error - {str(e)}"


def check_parameter_combinations(parameters: Dict[str, Any], model_type: str) -> List[str]:
    """Check for problematic parameter combinations."""
    issues = []
    
    try:
        # XGBoost specific checks
        if model_type == 'xgboost':
            if 'learning_rate' in parameters and 'n_estimators' in parameters:
                lr = parameters['learning_rate']
                n_est = parameters['n_estimators']
                if lr > 0.3 and n_est > 200:
                    issues.append("High learning rate with many estimators may cause overfitting")
                elif lr < 0.01 and n_est < 100:
                    issues.append("Low learning rate with few estimators may underfit")
        
        # Random Forest specific checks
        elif model_type == 'random_forest':
            if 'n_estimators' in parameters and 'max_depth' in parameters:
                n_est = parameters['n_estimators']
                max_depth = parameters['max_depth']
                if n_est > 500 and max_depth is None:
                    issues.append("Many estimators with unlimited depth may be slow and overfit")
        
        # SVR specific checks
        elif model_type == 'svr':
            if 'C' in parameters and 'gamma' in parameters:
                C = parameters['C']
                gamma = parameters['gamma']
                if isinstance(gamma, float) and C > 10 and gamma > 1:
                    issues.append("High C and gamma values may cause severe overfitting")
    
    except Exception as e:
        issues.append(f"Error checking parameter combinations: {str(e)}")
    
    return issues


def suggest_parameter_optimizations(model_type: str, parameters: Dict[str, Any], target_days: int) -> List[str]:
    """Suggest parameter optimizations based on model type and context."""
    suggestions = []
    
    try:
        schema = PARAMETER_SCHEMAS.get(model_type, {})
        ranges = schema.get('ranges', {})
        
        for param_name, param_value in parameters.items():
            if param_name in ranges:
                param_ranges = ranges[param_name]
                
                # Check if parameter is at extremes
                if isinstance(param_ranges, list):
                    min_val = min(param_ranges)
                    max_val = max(param_ranges)
                    
                    if param_value == min_val:
                        suggestions.append(f"Consider increasing {param_name} from {param_value} for better performance")
                    elif param_value == max_val:
                        suggestions.append(f"Consider decreasing {param_name} from {param_value} to avoid overfitting")
        
        # Context-specific suggestions
        if target_days > 7:
            if model_type in ['xgboost', 'gradient_boosting']:
                suggestions.append("For longer prediction horizons, consider reducing learning_rate and increasing n_estimators")
        
        if target_days == 1:
            if model_type == 'random_forest':
                suggestions.append("For daily predictions, consider allowing deeper trees (max_depth=None)")
    
    except Exception as e:
        suggestions.append(f"Error generating suggestions: {str(e)}")
    
    return suggestions


def get_context_recommendations(model_type: str, target_days: int) -> List[str]:
    """Get context-specific recommendations."""
    recommendations = []
    
    try:
        if target_days <= 3:
            recommendations.append("Short-term predictions: Consider ensemble methods for robustness")
        elif target_days <= 14:
            recommendations.append("Medium-term predictions: Balance between bias and variance")
        else:
            recommendations.append("Long-term predictions: Focus on preventing overfitting")
        
        if model_type in ['xgboost', 'gradient_boosting']:
            recommendations.append("Gradient boosting: Monitor for overfitting with early stopping")
        elif model_type == 'random_forest':
            recommendations.append("Random Forest: Generally robust, good default choice")
        elif model_type == 'svr':
            recommendations.append("SVR: Ensure features are properly scaled")
    
    except Exception as e:
        recommendations.append(f"Error generating context recommendations: {str(e)}")
    
    return recommendations


def format_parameter_summary(parameters: Dict[str, Any], schema: Dict[str, Any]) -> str:
    """Format parameters into a readable summary."""
    try:
        lines = []
        defaults = schema.get('defaults', {})
        
        for param_name, param_value in parameters.items():
            default_value = defaults.get(param_name, 'N/A')
            status = "✓ Custom" if param_value != default_value else "Default"
            lines.append(f"  • {param_name}: {param_value} ({status})")
        
        return "\\n".join(lines) if lines else "  • No parameters specified"
        
    except Exception as e:
        return f"  • Error formatting parameters: {str(e)}"


def assess_parameter_risk(parameters: Dict[str, Any], model_type: str) -> str:
    """Assess the risk level of the parameter configuration."""
    try:
        risk_factors = 0
        
        # Check for overfitting risks
        if model_type == 'xgboost':
            if parameters.get('learning_rate', 0.1) > 0.2:
                risk_factors += 1
            if parameters.get('max_depth', 6) > 10:
                risk_factors += 1
        
        elif model_type == 'random_forest':
            if parameters.get('max_depth') is None and parameters.get('min_samples_split', 2) <= 2:
                risk_factors += 1
        
        elif model_type == 'svr':
            if parameters.get('C', 1.0) > 10:
                risk_factors += 1
        
        if risk_factors == 0:
            return "Low (conservative parameters)"
        elif risk_factors == 1:
            return "Medium (some overfitting risk)"
        else:
            return "High (multiple overfitting risks)"
    
    except:
        return "Unknown"


def predict_performance_level(parameters: Dict[str, Any], model_type: str) -> str:
    """Predict expected performance level based on parameters."""
    try:
        schema = PARAMETER_SCHEMAS.get(model_type, {})
        defaults = schema.get('defaults', {})
        
        # Simple heuristic based on parameter choices
        custom_params = sum(1 for k, v in parameters.items() if defaults.get(k) != v)
        
        if custom_params == 0:
            return "Baseline (using default parameters)"
        elif custom_params <= 2:
            return "Good (some optimization)"
        else:
            return "Potentially High (custom tuning)"
    
    except:
        return "Unknown"


def rank_models_for_context(context: str, target_days: int, available_features: int, computational_budget: str) -> List[str]:
    """Rank models based on context and constraints."""
    try:
        context_info = MODELING_CONTEXTS.get(context, {})
        preferred_models = context_info.get('preferred_models', [])
        
        # Start with context preferences
        rankings = preferred_models.copy()
        
        # Add other models based on constraints
        all_models = ['xgboost', 'random_forest', 'gradient_boosting', 'svr', 'ridge_regression', 'extra_trees']
        for model in all_models:
            if model not in rankings:
                rankings.append(model)
        
        # Adjust based on computational budget
        if computational_budget == 'low':
            # Prefer faster models
            fast_models = ['ridge_regression', 'random_forest', 'extra_trees']
            rankings = [m for m in fast_models if m in rankings] + [m for m in rankings if m not in fast_models]
        
        return rankings[:6]  # Return top 6
        
    except Exception as e:
        # Fallback ranking
        return ['random_forest', 'xgboost', 'gradient_boosting', 'svr', 'ridge_regression', 'extra_trees']


def get_recommended_parameters(model_type: str, context: str, target_days: int, computational_budget: str) -> Dict[str, Any]:
    """Get recommended parameters for a model type and context."""
    try:
        schema = PARAMETER_SCHEMAS.get(model_type, {})
        defaults = schema.get('defaults', {}).copy()
        
        # Adjust based on context and constraints
        if model_type == 'xgboost':
            if target_days > 7:
                defaults['learning_rate'] = 0.05
                defaults['n_estimators'] = 200
            if computational_budget == 'low':
                defaults['n_estimators'] = 50
        
        elif model_type == 'random_forest':
            if target_days > 7:
                defaults['n_estimators'] = 200
            if computational_budget == 'low':
                defaults['n_estimators'] = 50
        
        return defaults
        
    except Exception as e:
        return {}


def get_model_rationale(model_type: str, context: str) -> str:
    """Get rationale for why a model is recommended for a context."""
    rationales = {
        'xgboost': 'Excellent performance with automatic feature selection and built-in regularization',
        'random_forest': 'Robust, stable, and handles various data types well with good interpretability',
        'gradient_boosting': 'Strong sequential learning with good bias-variance tradeoff',
        'svr': 'Effective for non-linear patterns with proper kernel selection',
        'ridge_regression': 'Simple, fast, and provides good baseline with linear relationships',
        'extra_trees': 'Fast training with good performance and natural regularization'
    }
    
    return rationales.get(model_type, 'General-purpose model suitable for various scenarios')


def format_param_recommendations(params: Dict[str, Any]) -> str:
    """Format parameter recommendations as a string."""
    if not params:
        return "Use defaults"
    
    param_strs = []
    for k, v in list(params.items())[:3]:  # Show top 3
        param_strs.append(f"{k}={v}")
    
    return ", ".join(param_strs) + ("..." if len(params) > 3 else "")


def get_feature_recommendations(context: str, available_features: int) -> List[str]:
    """Get feature recommendations for a context."""
    context_info = MODELING_CONTEXTS.get(context, {})
    key_features = context_info.get('key_features', [])
    
    recommendations = []
    if key_features:
        recommendations.append(f"Prioritize: {', '.join(key_features[:3])}")
    
    if available_features < 5:
        recommendations.append("Consider adding more technical indicators for better model performance")
    elif available_features > 20:
        recommendations.append("Consider feature selection to reduce dimensionality and overfitting risk")
    
    return recommendations


def generate_context_assessments(context: str, target_days: int, available_features: int, computational_budget: str) -> List[str]:
    """Generate assessments based on context and constraints."""
    assessments = []
    
    assessments.append(f"Prediction complexity: {'High' if target_days > 14 else 'Medium' if target_days > 3 else 'Low'}")
    assessments.append(f"Feature richness: {'High' if available_features > 15 else 'Medium' if available_features > 8 else 'Low'}")
    assessments.append(f"Computational resources: {computational_budget.title()}")
    
    return assessments


def get_key_tuning_parameters(model_type: str) -> str:
    """Get key parameters to focus on for tuning."""
    key_params = {
        'xgboost': 'n_estimators, learning_rate, max_depth',
        'random_forest': 'n_estimators, max_depth',
        'gradient_boosting': 'n_estimators, learning_rate',
        'svr': 'C, gamma',
        'ridge_regression': 'alpha',
        'extra_trees': 'n_estimators'
    }
    
    return key_params.get(model_type, 'model-specific parameters')


def get_min_data_requirement(target_days: int) -> int:
    """Get minimum data requirement based on prediction horizon."""
    base_requirement = 50
    return base_requirement + (target_days * 10)  # More data for longer predictions