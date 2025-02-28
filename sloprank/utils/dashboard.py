"""
Dashboard generation for SlopRank results.
"""
import json
import pandas as pd
import webbrowser
import threading
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

from ..config import logger

def generate_dashboard(
        rankings_path=None,
        confidence_path=None,
        categories_path=None,
        graph_path=None,
        output_path=None
    ):
    """
    Generate an HTML dashboard for SlopRank results.
    
    Parameters:
    -----------
    rankings_path : Path or str
        Path to the rankings JSON file
    confidence_path : Path or str
        Path to the confidence stats JSON file
    categories_path : Path or str
        Path to the category rankings JSON file
    graph_path : Path or str
        Path to the graph visualization image
    output_path : Path or str
        Path to save the dashboard HTML file
    
    Returns:
    --------
    Path
        Path to the generated dashboard HTML file
    """
    if rankings_path is None:
        rankings_path = Path("results/rankings.json")
    else:
        rankings_path = Path(rankings_path)
    
    if output_path is None:
        output_path = Path("results/dashboard.html")
    else:
        output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load rankings data
    with open(rankings_path, 'r') as f:
        rankings_data = json.load(f)
    
    # Load confidence data if available
    has_confidence = confidence_path is not None and Path(confidence_path).exists()
    confidence_data = None
    if has_confidence:
        with open(confidence_path, 'r') as f:
            confidence_data = json.load(f)
    
    # Load category rankings if available
    has_categories = categories_path is not None and Path(categories_path).exists()
    category_data = None
    if has_categories:
        with open(categories_path, 'r') as f:
            category_data = json.load(f)
    
    # Check if graph visualization is available
    has_graph = graph_path is not None and Path(graph_path).exists()
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SlopRank Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .bar-container {{
                width: 300px;
                background-color: #eee;
                border-radius: 4px;
                position: relative;
            }}
            .bar {{
                height: 20px;
                background-color: #4CAF50;
                border-radius: 4px;
            }}
            .error-bar {{
                position: absolute;
                height: 20px;
                background-color: rgba(0,0,0,0.2);
                z-index: 1;
            }}
            .image-container {{
                margin-top: 20px;
                text-align: center;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metadata {{
                font-size: 0.8em;
                color: #666;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SlopRank Dashboard</h1>
            
            <h2>Model Rankings</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Score</th>
                    <th>Visualization</th>
    """
    
    if has_confidence:
        html += """
                    <th>Confidence Interval</th>
        """
    
    html += """
                </tr>
    """
    
    # Add rows for each model
    if isinstance(rankings_data['rankings'][0], list):
        # Old format with list of lists
        ranked_items = rankings_data["rankings"]
        max_score = max([score for _, score in ranked_items])
    else:
        # New format with list of dicts
        ranked_items = [(item["model"], item["score"]) for item in rankings_data["rankings"]]
        max_score = max([item["score"] for item in rankings_data["rankings"]])
    
    for i, (model, score) in enumerate(ranked_items):
        bar_width = int(300 * score / max_score)
        confidence_html = ""
        
        if has_confidence and model in confidence_data["confidence_intervals"]:
            ci = confidence_data["confidence_intervals"][model]
            lower_pct = int(300 * ci["lower_bound"] / max_score)
            upper_pct = int(300 * ci["upper_bound"] / max_score)
            mean_pct = int(300 * ci["mean"] / max_score)
            
            confidence_html = f"""
                <td>
                    <div class="bar-container">
                        <div class="bar" style="width: {mean_pct}px;"></div>
                        <div class="error-bar" style="left: {lower_pct}px; width: {upper_pct - lower_pct}px;"></div>
                    </div>
                    {ci["mean"]:.6f} [{ci["lower_bound"]:.6f}, {ci["upper_bound"]:.6f}]
                </td>
            """
        
        html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{model}</td>
                    <td>{score:.6f}</td>
                    <td>
                        <div class="bar-container">
                            <div class="bar" style="width: {bar_width}px;"></div>
                        </div>
                    </td>
                    {confidence_html}
                </tr>
        """
    
    html += """
            </table>
    """
    
    # Add statistical significance if available
    if has_confidence and confidence_data.get("significance"):
        html += """
            <h2>Statistical Significance</h2>
            <table>
                <tr>
                    <th>Comparison</th>
                    <th>Significance</th>
                </tr>
        """
        
        for pair, is_significant in confidence_data["significance"].items():
            significance_str = "Significant" if is_significant else "Not significant"
            html += f"""
                <tr>
                    <td>{pair}</td>
                    <td>{significance_str}</td>
                </tr>
            """
        
        html += """
            </table>
        """
    
    # Add category rankings if available
    if has_categories and category_data:
        html += """
            <h2>Rankings by Category</h2>
        """
        
        for category, models in sorted(category_data.items()):
            max_score = max([item["score"] for item in models])
            
            html += f"""
                <h3>{category}</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Score</th>
                        <th>Visualization</th>
                    </tr>
            """
            
            for i, item in enumerate(models):
                model = item["model"]
                score = item["score"]
                bar_width = int(300 * score / max_score)
                
                html += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{model}</td>
                        <td>{score:.4f}</td>
                        <td>
                            <div class="bar-container">
                                <div class="bar" style="width: {bar_width}px;"></div>
                            </div>
                        </td>
                    </tr>
                """
            
            html += """
                </table>
            """
    
    # Add graph visualization if available
    if has_graph:
        rel_path = str(Path(graph_path).relative_to(Path.cwd()))
        html += f"""
            <h2>Endorsement Graph</h2>
            <div class="image-container">
                <img src="{rel_path}" alt="Endorsement Graph">
            </div>
        """
    
    # Add metadata
    html += f"""
            <div class="metadata">
                <p>Generated with SlopRank v{rankings_data['metadata'].get('version', '0.2.1')}</p>
                <p>Timestamp: {rankings_data['metadata'].get('timestamp', '')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML to file
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Dashboard generated at {output_path}")
    return output_path


def start_dashboard(dashboard_path=None, port=8000, open_browser=True):
    """
    Start a web server to view the SlopRank dashboard.
    
    Parameters:
    -----------
    dashboard_path : Path or str
        Path to the dashboard HTML file
    port : int
        Port for the web server
    open_browser : bool
        Whether to open a browser window automatically
    
    Returns:
    --------
    HTTPServer
        The server instance
    """
    if dashboard_path is None:
        dashboard_path = Path("results/dashboard.html")
    else:
        dashboard_path = Path(dashboard_path)
    
    if not dashboard_path.exists():
        logger.error(f"Dashboard file not found: {dashboard_path}")
        return None
    
    # Start server
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    url = f"http://localhost:{port}/{dashboard_path}"
    logger.info(f"Server started at {url}")
    
    # Open browser
    if open_browser:
        webbrowser.open(url)
    
    return httpd


if __name__ == "__main__":
    # Run as a standalone script
    dashboard_path = generate_dashboard()
    httpd = start_dashboard(dashboard_path)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        httpd.shutdown()