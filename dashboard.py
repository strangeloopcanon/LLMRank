#!/usr/bin/env python3
import json
import pandas as pd
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time

def generate_html():
    # Load rankings data
    rankings_path = Path("results/rankings.json")
    with open(rankings_path, 'r') as f:
        rankings_data = json.load(f)
    
    # Load confidence data if available
    confidence_path = Path("results/confidence_stats.json")
    has_confidence = confidence_path.exists()
    confidence_data = None
    if has_confidence:
        with open(confidence_path, 'r') as f:
            confidence_data = json.load(f)
    
    # Load category rankings if available
    category_path = Path("results/category_rankings.json")
    has_categories = category_path.exists()
    category_data = None
    if has_categories:
        with open(category_path, 'r') as f:
            category_data = json.load(f)
    
    # Generate HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SlopRank Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1, h2 {
                color: #333;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .bar-container {
                width: 300px;
                background-color: #eee;
                border-radius: 4px;
            }
            .bar {
                height: 20px;
                background-color: #4CAF50;
                border-radius: 4px;
            }
            .error-bar {
                position: relative;
                height: 20px;
                width: 10px;
                background-color: rgba(0,0,0,0.1);
            }
            .image-container {
                margin-top: 20px;
                text-align: center;
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
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
    max_score = max([entry[1] for entry in rankings_data["rankings"]])
    
    for i, (model, score) in enumerate(rankings_data["rankings"]):
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
                        <div class="error-bar" style="position: absolute; top: 0; left: {lower_pct}px; width: {upper_pct - lower_pct}px; height: 20px; background-color: rgba(0,0,0,0.2);"></div>
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
    graph_image_path = Path("results/visualizations/endorsement_graph.png")
    if graph_image_path.exists():
        html += """
            <h2>Endorsement Graph</h2>
            <div class="image-container">
                <img src="results/visualizations/endorsement_graph.png" alt="Endorsement Graph">
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML to file
    dashboard_path = Path("results/dashboard.html")
    with open(dashboard_path, 'w') as f:
        f.write(html)
    
    return dashboard_path

def start_server(port=8000):
    # Start HTTP server
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    print(f"Server started at http://localhost:{port}")
    return httpd

if __name__ == "__main__":
    dashboard_path = generate_html()
    print(f"Dashboard HTML generated at {dashboard_path}")
    
    port = 8000
    httpd = start_server(port)
    
    # Open browser
    url = f"http://localhost:{port}/results/dashboard.html"
    print(f"Opening dashboard at {url}")
    webbrowser.open(url)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down server...")
        httpd.shutdown()