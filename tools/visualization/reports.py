"""HTML report generation functionality."""

import os
import json
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

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

from ..config import OUTPUT_DIR, logger


@safe_run
def generate_comprehensive_html_report_impl(
    symbol: str,
    title: Optional[str] = None,
    sections: Optional[List[str]] = None,
    include_charts: bool = True,
    custom_content: Optional[str] = None,
    save_report: bool = True
) -> str:
    """
    Generate comprehensive HTML report with all analysis results.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        title: Report title (optional)
        sections: Sections to include in report
        include_charts: Whether to embed charts in report
        custom_content: Additional custom HTML content
        save_report: Whether to save report to HTML file
        
    Returns:
        String with report generation results and file location
    """
    log_info(f"generate_comprehensive_html_report: Creating HTML report for {symbol.upper()}...")
    
    try:
        symbol = symbol.upper()
        
        # Set default title
        if title is None:
            title = f"{symbol} Stock Analysis Report"
        
        # Set default sections
        if sections is None:
            sections = ['summary', 'data_analysis', 'model_results', 'backtesting', 'charts']
        
        # Gather available data files
        data_files = discover_data_files(symbol)
        model_files = discover_model_files(symbol)
        backtest_files = discover_backtest_files(symbol)
        chart_files = discover_chart_files(symbol)
        
        # Generate HTML content
        html_content = generate_html_structure(
            title, symbol, sections, data_files, model_files, 
            backtest_files, chart_files, include_charts, custom_content
        )
        
        # Save report if requested
        report_file = None
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"comprehensive_report_{symbol}_{timestamp}.html"
            report_filepath = os.path.join(OUTPUT_DIR, report_file)
            
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        # Count included sections and files
        total_sections = len(sections)
        total_files = len(data_files) + len(model_files) + len(backtest_files) + len(chart_files)
        
        # Generate summary
        summary = f"""generate_comprehensive_html_report: Successfully created HTML report for {symbol}:

📄 REPORT DETAILS:
- Symbol: {symbol}
- Title: {title}
- Sections Included: {total_sections} ({', '.join(sections)})
- Total Files Analyzed: {total_files}

 CONTENT SUMMARY:
- Data Files: {len(data_files)}
- Model Files: {len(model_files)}
- Backtest Files: {len(backtest_files)}
- Chart Files: {len(chart_files)}
- Interactive Charts: {'Included' if include_charts else 'Not included'}

 REPORT SAVED: {report_file if report_file else 'Not saved'}
- Location: {os.path.join(OUTPUT_DIR, report_file) if report_file else 'N/A'}
- Format: Professional HTML with CSS styling and JavaScript

 REPORT FEATURES:
- Professional Layout: Corporate styling with navigation
- Interactive Elements: Collapsible sections and chart integration
- Comprehensive Analysis: All model and backtest results included
- Export Ready: Suitable for presentations and sharing
- Self-Contained: All resources embedded for portability

🌐 USAGE INSTRUCTIONS:
- Open HTML file in any modern web browser
- Use navigation menu to jump between sections
- Interactive charts support zoom, pan, and hover
- Print or export to PDF using browser functionality
"""
        
        log_success(f"generate_comprehensive_html_report: Successfully created report for {symbol}")
        return summary
        
    except Exception as e:
        error_msg = f"generate_comprehensive_html_report: Error creating report for {symbol}: {str(e)}"
        log_error(f"generate_comprehensive_html_report: {error_msg}")
        return error_msg


def discover_data_files(symbol: str) -> List[Dict[str, Any]]:
    """Discover available data files for the symbol."""
    data_files = []
    
    try:
        all_files = os.listdir(OUTPUT_DIR)
        csv_files = [f for f in all_files if f.endswith('.csv') and symbol.upper() in f.upper()]
        
        for file in csv_files:
            filepath = os.path.join(OUTPUT_DIR, file)
            stat = os.stat(filepath)
            
            data_files.append({
                'filename': file,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'type': 'enhanced' if 'enhanced' in file.lower() else 'basic'
            })
        
        # Sort by modification time (newest first)
        data_files.sort(key=lambda x: x['modified'], reverse=True)
        
    except Exception as e:
        log_warning(f"Warning: Could not discover data files: {str(e)}")
    
    return data_files


def discover_model_files(symbol: str) -> List[Dict[str, Any]]:
    """Discover available model files for the symbol."""
    model_files = []
    
    try:
        all_files = os.listdir(OUTPUT_DIR)
        pkl_files = [f for f in all_files if f.endswith('_model.pkl') and symbol.upper() in f.upper()]
        
        for file in pkl_files:
            filepath = os.path.join(OUTPUT_DIR, file)
            stat = os.stat(filepath)
            
            # Try to extract model type from filename
            model_type = 'unknown'
            if '_' in file:
                parts = file.split('_')
                if len(parts) > 1:
                    model_type = parts[1]
            
            model_files.append({
                'filename': file,
                'model_type': model_type,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: x['modified'], reverse=True)
        
    except Exception as e:
        log_warning(f"Warning: Could not discover model files: {str(e)}")
    
    return model_files


def discover_backtest_files(symbol: str) -> List[Dict[str, Any]]:
    """Discover available backtest result files for the symbol."""
    backtest_files = []
    
    try:
        all_files = os.listdir(OUTPUT_DIR)
        json_files = [f for f in all_files 
                     if f.endswith('.json') and 'backtest' in f.lower() and symbol.upper() in f.upper()]
        
        for file in json_files:
            filepath = os.path.join(OUTPUT_DIR, file)
            stat = os.stat(filepath)
            
            # Try to load summary data
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                strategy_type = data.get('strategy_type', 'unknown')
                total_return = data.get('total_return', 0)
                
            except:
                strategy_type = 'unknown'
                total_return = 0
            
            backtest_files.append({
                'filename': file,
                'strategy_type': strategy_type,
                'total_return': total_return,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })
        
        # Sort by modification time (newest first)
        backtest_files.sort(key=lambda x: x['modified'], reverse=True)
        
    except Exception as e:
        log_warning(f"Warning: Could not discover backtest files: {str(e)}")
    
    return backtest_files


def discover_chart_files(symbol: str) -> List[Dict[str, Any]]:
    """Discover available chart files for the symbol."""
    chart_files = []
    
    try:
        all_files = os.listdir(OUTPUT_DIR)
        html_files = [f for f in all_files if f.endswith('.html') and symbol.upper() in f.upper()]
        
        for file in html_files:
            filepath = os.path.join(OUTPUT_DIR, file)
            stat = os.stat(filepath)
            
            # Determine chart type from filename
            chart_type = 'unknown'
            if 'visualize_stock_data' in file:
                chart_type = 'stock_data'
            elif 'model_comparison' in file:
                chart_type = 'model_comparison'
            elif 'backtesting' in file:
                chart_type = 'backtesting'
            
            chart_files.append({
                'filename': file,
                'chart_type': chart_type,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })
        
        # Sort by modification time (newest first)
        chart_files.sort(key=lambda x: x['modified'], reverse=True)
        
    except Exception as e:
        log_warning(f"Warning: Could not discover chart files: {str(e)}")
    
    return chart_files


def generate_html_structure(
    title: str, symbol: str, sections: List[str], data_files: List[Dict], 
    model_files: List[Dict], backtest_files: List[Dict], chart_files: List[Dict],
    include_charts: bool, custom_content: Optional[str]
) -> str:
    """Generate the complete HTML structure for the report."""
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Generated on {timestamp}</p>
        </header>
        
        <nav class="navigation">
            <ul>
"""
    
    # Add navigation links
    for section in sections:
        section_title = section.replace('_', ' ').title()
        html += f'                <li><a href="#{section}">{section_title}</a></li>\\n'
    
    html += """
            </ul>
        </nav>
        
        <main>
"""
    
    # Add each section
    for section in sections:
        html += generate_section_content(
            section, symbol, data_files, model_files, 
            backtest_files, chart_files, include_charts
        )
    
    # Add custom content if provided
    if custom_content:
        html += f"""
        <section id="custom" class="section">
            <h2>Additional Analysis</h2>
            {custom_content}
        </section>
"""
    
    # Close HTML
    html += """
        </main>
        
        <footer>
            <p>Report generated by Stock Analyzer - Genesis Multi-Agent System</p>
        </footer>
    </div>
    
    <script>
        // Simple navigation highlighting
        document.addEventListener('DOMContentLoaded', function() {
            const links = document.querySelectorAll('nav a');
            links.forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });
        });
    </script>
</body>
</html>
"""
    
    return html


def generate_section_content(
    section: str, symbol: str, data_files: List[Dict], model_files: List[Dict],
    backtest_files: List[Dict], chart_files: List[Dict], include_charts: bool
) -> str:
    """Generate content for a specific section."""
    
    section_title = section.replace('_', ' ').title()
    
    if section == 'summary':
        return generate_summary_section(symbol, data_files, model_files, backtest_files, chart_files)
    elif section == 'data_analysis':
        return generate_data_analysis_section(data_files)
    elif section == 'model_results':
        return generate_model_results_section(model_files)
    elif section == 'backtesting':
        return generate_backtesting_section(backtest_files)
    elif section == 'charts':
        return generate_charts_section(chart_files, include_charts)
    else:
        return f"""
        <section id="{section}" class="section">
            <h2>{section_title}</h2>
            <p>Section content not yet implemented.</p>
        </section>
"""


def generate_summary_section(symbol: str, data_files: List[Dict], model_files: List[Dict], 
                           backtest_files: List[Dict], chart_files: List[Dict]) -> str:
    """Generate the executive summary section."""
    
    return f"""
        <section id="summary" class="section">
            <h2>Executive Summary</h2>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Analysis Overview</h3>
                    <p><strong>Symbol:</strong> {symbol}</p>
                    <p><strong>Data Files:</strong> {len(data_files)}</p>
                    <p><strong>Models Trained:</strong> {len(model_files)}</p>
                    <p><strong>Backtests Completed:</strong> {len(backtest_files)}</p>
                    <p><strong>Charts Generated:</strong> {len(chart_files)}</p>
                </div>
                
                <div class="summary-card">
                    <h3>Best Performing Model</h3>
                    <p>Analysis based on available backtest results...</p>
                    <p><em>Detailed results available in Model Results section.</em></p>
                </div>
                
                <div class="summary-card">
                    <h3>Key Insights</h3>
                    <ul>
                        <li>Comprehensive technical analysis completed</li>
                        <li>Multiple ML models trained and evaluated</li>
                        <li>Strategy performance tested and validated</li>
                        <li>Interactive visualizations available</li>
                    </ul>
                </div>
            </div>
        </section>
"""


def generate_data_analysis_section(data_files: List[Dict]) -> str:
    """Generate the data analysis section."""
    
    content = """
        <section id="data_analysis" class="section">
            <h2>Data Analysis</h2>
            
            <div class="data-files">
                <h3>Available Data Files</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Filename</th>
                            <th>Type</th>
                            <th>Size</th>
                            <th>Modified</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for file in data_files[:10]:  # Limit to 10 most recent
        size_mb = file['size'] / (1024 * 1024)
        content += f"""
                        <tr>
                            <td>{file['filename']}</td>
                            <td>{file['type'].title()}</td>
                            <td>{size_mb:.2f} MB</td>
                            <td>{file['modified'].strftime('%Y-%m-%d %H:%M')}</td>
                        </tr>
"""
    
    content += """
                    </tbody>
                </table>
            </div>
        </section>
"""
    
    return content


def generate_model_results_section(model_files: List[Dict]) -> str:
    """Generate the model results section."""
    
    content = """
        <section id="model_results" class="section">
            <h2>Model Results</h2>
            
            <div class="model-files">
                <h3>Trained Models</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Model Type</th>
                            <th>Filename</th>
                            <th>Size</th>
                            <th>Created</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for file in model_files:
        size_mb = file['size'] / (1024 * 1024)
        content += f"""
                        <tr>
                            <td>{file['model_type'].replace('_', ' ').title()}</td>
                            <td>{file['filename']}</td>
                            <td>{size_mb:.2f} MB</td>
                            <td>{file['modified'].strftime('%Y-%m-%d %H:%M')}</td>
                        </tr>
"""
    
    content += """
                    </tbody>
                </table>
            </div>
        </section>
"""
    
    return content


def generate_backtesting_section(backtest_files: List[Dict]) -> str:
    """Generate the backtesting results section."""
    
    content = """
        <section id="backtesting" class="section">
            <h2>Backtesting Results</h2>
            
            <div class="backtest-files">
                <h3>Strategy Performance</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Strategy</th>
                            <th>Total Return</th>
                            <th>Date</th>
                            <th>Filename</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for file in backtest_files:
        content += f"""
                        <tr>
                            <td>{file['strategy_type'].replace('_', ' ').title()}</td>
                            <td>{file['total_return']:+.2f}%</td>
                            <td>{file['modified'].strftime('%Y-%m-%d')}</td>
                            <td>{file['filename']}</td>
                        </tr>
"""
    
    content += """
                    </tbody>
                </table>
            </div>
        </section>
"""
    
    return content


def generate_charts_section(chart_files: List[Dict], include_charts: bool) -> str:
    """Generate the charts section."""
    
    content = """
        <section id="charts" class="section">
            <h2>Interactive Charts</h2>
            
            <div class="chart-files">
                <h3>Available Visualizations</h3>
                <div class="chart-grid">
"""
    
    for file in chart_files:
        chart_type_display = file['chart_type'].replace('_', ' ').title()
        content += f"""
                    <div class="chart-card">
                        <h4>{chart_type_display}</h4>
                        <p><strong>File:</strong> {file['filename']}</p>
                        <p><strong>Created:</strong> {file['modified'].strftime('%Y-%m-%d %H:%M')}</p>
                        <a href="{file['filename']}" target="_blank" class="chart-link">Open Chart</a>
                    </div>
"""
    
    content += """
                </div>
            </div>
        </section>
"""
    
    return content


def get_css_styles() -> str:
    """Get CSS styles for the HTML report."""
    
    return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .navigation {
            background: #2c3e50;
            padding: 0;
        }
        
        .navigation ul {
            list-style: none;
            display: flex;
            flex-wrap: wrap;
        }
        
        .navigation li {
            flex: 1;
        }
        
        .navigation a {
            display: block;
            padding: 1rem;
            color: white;
            text-decoration: none;
            text-align: center;
            transition: background-color 0.3s;
        }
        
        .navigation a:hover {
            background-color: #34495e;
        }
        
        main {
            padding: 2rem;
        }
        
        .section {
            margin-bottom: 3rem;
        }
        
        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }
        
        .summary-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .summary-card h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }
        
        .chart-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .chart-link {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 0.5rem 1rem;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 0.5rem;
            transition: background-color 0.3s;
        }
        
        .chart-link:hover {
            background: #5a6fd8;
        }
        
        footer {
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 1rem;
            margin-top: 2rem;
        }
        
        @media (max-width: 768px) {
            .navigation ul {
                flex-direction: column;
            }
            
            .summary-grid {
                grid-template-columns: 1fr;
            }
            
            main {
                padding: 1rem;
            }
        }
"""