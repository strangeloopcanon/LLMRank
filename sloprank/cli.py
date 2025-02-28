import logging
import json
import threading
from pathlib import Path
from typing import Dict, List

import click
import pandas as pd

from .collect import collect_raw_evaluations, collect_responses
from .config import DEFAULT_CONFIG, EvalConfig, VisualizationConfig, ConfidenceConfig, WebDashboardConfig, logger
from .parse import parse_evaluation_rows
from .rank import (
    build_endorsement_graph, 
    compute_pagerank, 
    compute_categorical_pageranks,
    finalize_rankings
)

# Try importing dashboard libraries
try:
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_DASH = True
except ImportError:
    logger.warning("Dash not found. Web dashboard will be disabled.")
    HAS_DASH = False

def categorize_prompts(prompts_df: pd.DataFrame, config: EvalConfig) -> Dict[str, List[str]]:
    """
    Process the prompts DataFrame to extract categories.
    If a 'Category' column exists, use it to categorize prompts.
    Otherwise, try to infer categories using keyword matching.
    """
    categories = {}
    
    if 'Category' in prompts_df.columns:
        # Use explicit categories from the prompts file
        for category in prompts_df['Category'].unique():
            if pd.notna(category) and category:
                category_prompts = prompts_df[prompts_df['Category'] == category]['Questions'].tolist()
                categories[category] = category_prompts
    elif config.prompt_categories:
        # Use categories from the configuration
        return config.prompt_categories
    else:
        # Try to infer categories using keywords (basic implementation)
        # In a real implementation, you might use NLP techniques or clustering
        keywords = {
            'reasoning': ['reason', 'logic', 'why', 'how', 'explain', 'analyze'],
            'creativity': ['creative', 'imagine', 'story', 'design', 'invent'],
            'knowledge': ['fact', 'define', 'what is', 'history', 'science'],
            'coding': ['code', 'function', 'algorithm', 'program', 'script'],
        }
        
        # Initialize categories
        for category in keywords:
            categories[category] = []
        
        # Categorize prompts based on keywords
        for prompt in prompts_df['Questions'].tolist():
            categorized = False
            prompt_lower = prompt.lower()
            
            for category, terms in keywords.items():
                if any(term in prompt_lower for term in terms):
                    categories[category].append(prompt)
                    categorized = True
                    break
            
            if not categorized:
                if 'uncategorized' not in categories:
                    categories['uncategorized'] = []
                categories['uncategorized'].append(prompt)
    
    # Only keep categories with prompts
    return {k: v for k, v in categories.items() if v}

def start_dashboard(config: EvalConfig, rankings_path: Path):
    """
    Start a Dash web dashboard for interactive visualization.
    """
    if not HAS_DASH or not config.web_dashboard.enabled:
        return
    
    try:
        # Load rankings data
        with open(rankings_path, 'r') as f:
            data = json.load(f)
        
        # Create Dash app
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div([
            html.H1("SlopRank Dashboard"),
            
            html.Div([
                html.H2("Model Rankings"),
                dcc.Graph(
                    id='ranking-graph',
                    figure={
                        'data': [
                            {'x': [item['model'] for item in data['rankings']], 
                             'y': [item['score'] for item in data['rankings']], 
                             'type': 'bar', 'name': 'PageRank Score'}
                        ],
                        'layout': {
                            'title': 'Model PageRank Scores',
                            'xaxis': {'title': 'Model'},
                            'yaxis': {'title': 'PageRank Score'}
                        }
                    }
                )
            ]),
            
            # Add category rankings if available
            html.Div([
                html.H2("Rankings by Category"),
                html.Div([
                    html.Label("Select Category:"),
                    dcc.Dropdown(
                        id='category-dropdown',
                        options=[{'label': cat, 'value': cat} 
                                for cat in data.get('category_rankings', {}).keys()],
                        value=next(iter(data.get('category_rankings', {}).keys()), None)
                    )
                ]) if data.get('category_rankings') else html.Div("No category data available."),
                dcc.Graph(id='category-graph')
            ]) if data.get('category_rankings') else html.Div(),
            
            # Add confidence intervals if available
            html.Div([
                html.H2("Confidence Intervals"),
                dcc.Graph(
                    id='confidence-graph',
                    figure={
                        'data': [
                            {
                                'x': [model for model in data['confidence_intervals'].keys()],
                                'y': [stats['mean'] for stats in data['confidence_intervals'].values()],
                                'error_y': {
                                    'type': 'data',
                                    'symmetric': False,
                                    'array': [
                                        stats['upper_bound'] - stats['mean'] 
                                        for stats in data['confidence_intervals'].values()
                                    ],
                                    'arrayminus': [
                                        stats['mean'] - stats['lower_bound'] 
                                        for stats in data['confidence_intervals'].values()
                                    ]
                                },
                                'type': 'scatter',
                                'mode': 'markers',
                                'marker': {'size': 10}
                            }
                        ],
                        'layout': {
                            'title': '95% Confidence Intervals',
                            'xaxis': {'title': 'Model'},
                            'yaxis': {'title': 'PageRank Score'}
                        }
                    }
                )
            ]) if data.get('confidence_intervals') else html.Div(),
            
            # Add link to static visualizations
            html.Div([
                html.H2("Visualizations"),
                html.P([
                    "View the static graph visualization ",
                    html.A("here", href=f"/{config.output_dir}/visualizations/endorsement_graph.png", target="_blank"),
                    " or the interactive version ",
                    html.A("here", href=f"/{config.output_dir}/visualizations/endorsement_graph.html", target="_blank"),
                    "."
                ])
            ])
        ])
        
        # Define callbacks
        @app.callback(
            Output('category-graph', 'figure'),
            [Input('category-dropdown', 'value')]
        )
        def update_category_graph(selected_category):
            if not selected_category or not data.get('category_rankings'):
                return {}
                
            cat_data = data['category_rankings'].get(selected_category, [])
            return {
                'data': [
                    {'x': [item['model'] for item in cat_data], 
                    'y': [item['score'] for item in cat_data], 
                    'type': 'bar', 'name': 'PageRank Score'}
                ],
                'layout': {
                    'title': f'Model Rankings for Category: {selected_category}',
                    'xaxis': {'title': 'Model'},
                    'yaxis': {'title': 'PageRank Score'}
                }
            }
        
        # Run the server in a separate thread
        def run_server():
            app.run_server(
                host=config.web_dashboard.host,
                port=config.web_dashboard.port,
                debug=config.web_dashboard.debug
            )
        
        dashboard_thread = threading.Thread(target=run_server)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Print info message
        if config.web_dashboard.auto_open_browser:
            import webbrowser
            url = f"http://{config.web_dashboard.host}:{config.web_dashboard.port}"
            webbrowser.open(url)
            
        logger.info(f"Dashboard running at http://{config.web_dashboard.host}:{config.web_dashboard.port}")
        logger.info("Press Ctrl+C to exit")
        
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")


@click.group()
def cli():
    """SlopRank - Peer-based LLM cross-evaluation system."""
    pass


@cli.command()
@click.option("--prompts", default="prompts.xlsx", help="Path to prompts Excel file")
@click.option("--output-dir", default="results", help="Output directory for results")
@click.option("--models", help="Comma-separated list of models to evaluate")
@click.option(
    "--responses",
    help="Path to CSV of responses generated by a separate agent runner",
    default="",
)
@click.option(
    "--visualize/--no-visualize", 
    default=True, 
    help="Enable/disable graph visualization"
)
@click.option(
    "--interactive/--no-interactive", 
    default=True, 
    help="Enable/disable interactive visualization"
)
@click.option(
    "--confidence/--no-confidence", 
    default=True, 
    help="Enable/disable confidence interval calculation"
)
@click.option(
    "--dashboard/--no-dashboard", 
    default=False, 
    help="Enable/disable web dashboard"
)
@click.option(
    "--dashboard-port", 
    default=8050, 
    help="Port for web dashboard"
)
def run(prompts, output_dir, models, responses, visualize, interactive, confidence, dashboard, dashboard_port):
    """
    Run the full SlopRank evaluation workflow.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Parse model list from command line
    model_list = models.split(",") if models else None
    
    # Create visualization config
    vis_config = VisualizationConfig(
        enabled=visualize,
        interactive=interactive
    )
    
    # Create confidence config
    conf_config = ConfidenceConfig(
        enabled=confidence
    )
    
    # Create web dashboard config
    dash_config = WebDashboardConfig(
        enabled=dashboard,
        port=dashboard_port
    )
    
    # Create main config
    config = EvalConfig(
        model_names=model_list or DEFAULT_CONFIG.model_names,
        evaluation_method=1,  # numeric rating
        use_subset_evaluation=True,
        evaluators_subset_size=3,
        output_dir=Path(output_dir),
        visualization=vis_config,
        confidence=conf_config,
        web_dashboard=dash_config
    )
    logger.info(f"Using config: {config}")

    # 1a) If we generated the responses in another tool and are piping them
    # to SlopRank UNIX-style, we don't need to load/run the prompts
    if responses:
        responses_df = pd.read_csv(responses)
        prompts_df = pd.DataFrame({'Questions': responses_df['prompt'].unique()})
    else:
        # 1) Read prompts
        prompts_df = pd.read_excel(prompts)
        prompt_pairs = list(
            zip(
                prompts_df["Questions"].tolist(),
                prompts_df["Answer_key"].tolist()
                if "Answer_key" in prompts_df.columns
                else [None] * len(prompts_df),
            )
        )

        # 2) Collect responses
        responses_df = collect_responses(prompt_pairs, config)

    # Process prompt categories
    config.prompt_categories = categorize_prompts(prompts_df, config)
    if config.prompt_categories:
        logger.info(f"Found {len(config.prompt_categories)} prompt categories: {', '.join(config.prompt_categories.keys())}")

    # 3) Collect raw evaluations
    raw_eval_df = collect_raw_evaluations(responses_df, config)

    # 4) Parse evaluation rows
    eval_path = config.output_dir / "evaluations.csv"
    if eval_path.exists():
        logger.info(f"Loading existing parsed evaluations from {eval_path}")
        evaluations_df = pd.read_csv(eval_path)
    else:
        evaluations_df = parse_evaluation_rows(raw_eval_df, config)
        evaluations_df.to_csv(eval_path, index=False)
        logger.info(f"Saved parsed evaluations to {eval_path}")

    # 5) Build endorsement graph
    G = build_endorsement_graph(evaluations_df, config)
    
    # 6) Compute overall PageRank
    pagerank_scores = compute_pagerank(G)
    
    # 7) Compute category-specific PageRank scores if categories exist
    category_rankings = None
    if config.prompt_categories:
        category_rankings = compute_categorical_pageranks(G, config.prompt_categories)
    
    # 8) Finalize rankings and generate visualizations
    finalize_rankings(
        pagerank_scores, 
        config, 
        G=G,
        evaluations_df=evaluations_df,
        category_rankings=category_rankings
    )
    
    # 9) Start web dashboard if enabled
    if config.web_dashboard.enabled and HAS_DASH:
        rankings_path = config.output_dir / "rankings.json"
        if rankings_path.exists():
            start_dashboard(config, rankings_path)


@cli.command()
@click.option("--output-dir", default="results", help="Output directory containing results")
@click.option("--port", default=8050, help="Dashboard port")
def dashboard(output_dir, port):
    """
    Start the web dashboard for existing results.
    """
    if not HAS_DASH:
        logger.error("Dash not found. Please install with 'pip install dash plotly'")
        return
    
    config = EvalConfig(
        model_names=DEFAULT_CONFIG.model_names,
        evaluation_method=1,
        use_subset_evaluation=True,
        evaluators_subset_size=3,
        output_dir=Path(output_dir),
        web_dashboard=WebDashboardConfig(
            enabled=True,
            port=port,
            auto_open_browser=True
        )
    )
    
    rankings_path = Path(output_dir) / "rankings.json"
    if not rankings_path.exists():
        logger.error(f"Rankings file not found: {rankings_path}")
        return
    
    logger.info(f"Starting dashboard for results in {output_dir}")
    start_dashboard(config, rankings_path)
    
    # Keep the main thread alive
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")


def main():
    """Entry point for CLI."""
    # Register utility commands if available
    try:
        from .utils.commands import register_utils_commands
        register_utils_commands(cli)
    except ImportError:
        pass
    
    cli()
