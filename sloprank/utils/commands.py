"""
Command-line utilities for SlopRank.
"""
import click
import pandas as pd
import json
import threading
import time
from pathlib import Path
import webbrowser

from ..config import logger
from ..config import VisualizationConfig
from .visualization import generate_visualization

# Import confidence and dashboard modules if available
try:
    from .confidence import compute_confidence_intervals
    HAS_CONFIDENCE = True
except ImportError:
    HAS_CONFIDENCE = False

try:
    from .dashboard import generate_dashboard, start_dashboard
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False

# Import category analysis if available
try:
    from .categorization import categorize_prompts, analyze_categorized_evaluations
    HAS_CATEGORIES = True
except ImportError:
    HAS_CATEGORIES = False


@click.group()
def utils():
    """Utility commands for SlopRank."""
    pass


@utils.command()
@click.option("--rankings", default="results/rankings.json", help="Path to rankings JSON file")
@click.option("--evaluations", default="results/evaluations.csv", help="Path to evaluations CSV file")
@click.option("--output-dir", default="results/visualizations", help="Output directory for visualizations")
@click.option("--layout", default="spring", help="Graph layout [spring, circular, kamada_kawai, spectral]")
@click.option("--interactive/--no-interactive", default=True, help="Generate interactive HTML visualization")
def visualize(rankings, evaluations, output_dir, layout, interactive):
    """Generate visualizations for the SlopRank endorsement graph."""
    vis_config = VisualizationConfig(
        enabled=True,
        interactive=interactive,
        layout=layout
    )
    try:
        generate_visualization(
            rankings_path=rankings,
            evaluations_path=evaluations,
            output_dir=output_dir,
            vis_config=vis_config
        )
        click.echo(f"Visualizations generated in {output_dir}")
    except Exception as e:
        click.echo(f"Error generating visualizations: {e}", err=True)


@utils.command()
@click.option("--evaluations", default="results/evaluations.csv", help="Path to evaluations CSV file")
@click.option("--output", default="results/confidence_stats.json", help="Output file for confidence data")
@click.option("--iterations", default=500, help="Number of bootstrap iterations")
@click.option("--confidence-level", default=0.95, help="Confidence level (0.0-1.0)")
def confidence(evaluations, output, iterations, confidence_level):
    """Compute confidence intervals for SlopRank rankings."""
    if not HAS_CONFIDENCE:
        click.echo("Confidence module not available. Install numpy to use this feature.", err=True)
        return

    try:
        from .confidence import compute_confidence_intervals
        stats = compute_confidence_intervals(
            evaluations_path=evaluations,
            output_path=output,
            iterations=iterations,
            confidence_level=confidence_level
        )
        click.echo(f"Confidence statistics saved to {output}")
    except Exception as e:
        click.echo(f"Error computing confidence intervals: {e}", err=True)


@utils.command()
@click.option("--prompts", default="prompts.csv", help="Path to prompts Excel file")
@click.option("--evaluations", default="results/evaluations.csv", help="Path to evaluations CSV file")
@click.option("--output-dir", default="results", help="Output directory for category analysis")
def categorize(prompts, evaluations, output_dir):
    """Categorize prompts and analyze model performance by category."""
    if not HAS_CATEGORIES:
        click.echo("Categorization module not available.", err=True)
        return

    try:
        from .categorization import categorize_prompts, analyze_categorized_evaluations
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Categorize prompts
        categories = categorize_prompts(prompts_file=prompts)
        
        # Analyze performance by category
        analyze_categorized_evaluations(
            categorized_prompts=categories,
            evaluations_path=evaluations,
            output_dir=output_dir
        )
        
        click.echo(f"Category analysis saved to {output_dir / 'category_rankings.json'}")
    except Exception as e:
        click.echo(f"Error categorizing prompts: {e}", err=True)


@utils.command()
@click.option("--rankings", default="results/rankings.json", help="Path to rankings JSON file")
@click.option("--confidence", default="results/confidence_stats.json", help="Path to confidence stats JSON")
@click.option("--categories", default="results/category_rankings.json", help="Path to category rankings JSON")
@click.option("--graph", default="results/visualizations/endorsement_graph.png", help="Path to graph visualization")
@click.option("--output", default="results/dashboard.html", help="Output path for dashboard HTML")
def dashboard(rankings, confidence, categories, graph, output):
    """Generate HTML dashboard for SlopRank results."""
    if not HAS_DASHBOARD:
        click.echo("Dashboard module not available.", err=True)
        return

    try:
        from .dashboard import generate_dashboard
        
        dashboard_path = generate_dashboard(
            rankings_path=rankings,
            confidence_path=confidence if Path(confidence).exists() else None,
            categories_path=categories if Path(categories).exists() else None,
            graph_path=graph if Path(graph).exists() else None,
            output_path=output
        )
        
        click.echo(f"Dashboard generated at {dashboard_path}")
    except Exception as e:
        click.echo(f"Error generating dashboard: {e}", err=True)


@utils.command()
@click.option("--dashboard", default="results/dashboard.html", help="Path to dashboard HTML file")
@click.option("--port", default=8000, help="Port for the web server")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def serve(dashboard, port, no_browser):
    """Start a web server to view the SlopRank dashboard."""
    try:
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        
        dashboard_path = Path(dashboard)
        if not dashboard_path.exists():
            click.echo(f"Dashboard file not found: {dashboard_path}", err=True)
            return
        
        # Start server
        server_address = ('', port)
        httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        url = f"http://localhost:{port}/{dashboard}"
        click.echo(f"Server started at {url}")
        
        # Open browser
        if not no_browser:
            webbrowser.open(url)
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo("Shutting down server...")
            httpd.shutdown()
    
    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)


def register_utils_commands(cli):
    """Register utility commands with the main CLI."""
    cli.add_command(utils)