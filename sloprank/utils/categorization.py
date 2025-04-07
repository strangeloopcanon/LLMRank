"""
Prompt categorization and category-based analysis.
"""
import json
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

from ..config import logger

def categorize_prompts(prompts_file=None, save_categorized=True):
    """
    Read prompts from Excel file and automatically categorize them.
    If a 'Category' column exists, it will use those categories.
    Otherwise, it will attempt to infer categories based on content.
    
    Parameters:
    -----------
    prompts_file : Path or str
        Path to the prompts Excel file
    save_categorized : bool
        Whether to save the categorized prompts back to an Excel file
    
    Returns:
    --------
    dict
        Dictionary mapping category names to lists of prompts
    """
    if prompts_file is None:
        prompts_file = Path("prompts.csv")
    else:
        prompts_file = Path(prompts_file)
    
    logger.info(f"Reading prompts from {prompts_file}...")
    
    # Read prompts from Excel
    prompts_df = pd.read_csv(prompts_file)
    
    # Check if a Category column exists
    if 'Category' in prompts_df.columns:
        categories = defaultdict(list)
        
        # Group prompts by category
        for _, row in prompts_df.iterrows():
            if pd.notna(row['Category']) and row['Category']:
                categories[row['Category']].append(row['Questions'])
            else:
                if 'Uncategorized' not in categories:
                    categories['Uncategorized'] = []
                categories['Uncategorized'].append(row['Questions'])
        
        logger.info(f"Found {len(categories)} categories in the Excel file.")
    else:
        # Infer categories based on content
        categories = infer_categories(prompts_df['Questions'].tolist())
        
        if save_categorized:
            # Add inferred categories back to the DataFrame
            category_map = {}
            for category, prompts in categories.items():
                for prompt in prompts:
                    category_map[prompt] = category
            
            prompts_df['Category'] = prompts_df['Questions'].map(category_map)
            
            # Save the categorized DataFrame back to Excel
            output_path = prompts_file.with_stem(prompts_file.stem + "_categorized")
            prompts_df.to_csv(output_path, index=False)
            logger.info(f"Saved categorized prompts to {output_path}")
    
    # Return categories as a dictionary with lists of prompts
    return dict(categories)


def infer_categories(prompts):
    """
    Infer categories from prompt content using keyword matching.
    
    Parameters:
    -----------
    prompts : list
        List of prompts to categorize
    
    Returns:
    --------
    dict
        Dictionary mapping category names to lists of prompts
    """
    logger.info("Inferring categories from prompt content...")
    
    # Define category keywords
    keywords = {
        'Reasoning': ['reason', 'logic', 'why', 'how', 'explain', 'analyze', 'evaluate', 'assess', 'examine'],
        'Creativity': ['creative', 'imagine', 'story', 'design', 'invent', 'fiction', 'innovative'],
        'Knowledge': ['fact', 'define', 'what is', 'history', 'science', 'describe', 'information'],
        'Coding': ['code', 'function', 'algorithm', 'program', 'script', 'implementation'],
        'Opinion': ['opinion', 'believe', 'think', 'perspective', 'view', 'stance'],
        'Technical': ['technical', 'engineering', 'system', 'mechanism', 'process'],
        'Economic': ['economic', 'finance', 'market', 'money', 'business', 'trade', 'commerce', 'tax'],
        'Medical': ['medical', 'health', 'disease', 'treatment', 'cure', 'patient', 'doctor', 'hospital'],
        'Political': ['political', 'government', 'policy', 'regulation', 'law', 'legal'],
        'Ethical': ['ethical', 'moral', 'right', 'wrong', 'should', 'ethics', 'values'],
    }
    
    # Categorize prompts
    categories = defaultdict(list)
    
    for prompt in prompts:
        prompt_lower = prompt.lower()
        
        # Try to match prompt to a category
        matched = False
        for category, terms in keywords.items():
            if any(term in prompt_lower for term in terms):
                categories[category].append(prompt)
                matched = True
                break
        
        # If no match, add to Uncategorized
        if not matched:
            categories['Uncategorized'].append(prompt)
    
    # Count prompts per category
    for category, prompts in categories.items():
        logger.info(f"Category '{category}': {len(prompts)} prompts")
    
    return categories


def analyze_categorized_evaluations(
        categorized_prompts, 
        evaluations_path=None, 
        output_dir=None
    ):
    """
    Analyze evaluations based on prompt categories.
    
    Parameters:
    -----------
    categorized_prompts : dict
        Dictionary mapping category names to lists of prompts
    evaluations_path : Path or str
        Path to the evaluations CSV file
    output_dir : Path or str
        Directory to save the output files
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with category analysis results
    """
    if evaluations_path is None:
        evaluations_path = Path("results/evaluations.csv")
    else:
        evaluations_path = Path(evaluations_path)
    
    if output_dir is None:
        output_dir = Path("results")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluations
    logger.info(f"Loading evaluations from {evaluations_path}...")
    evals_df = pd.read_csv(evaluations_path)
    
    # Filter out failed evaluations
    evals_df = evals_df[evals_df["parse_failed"] == False]
    
    # Create a flat mapping of prompt -> category
    prompt_to_category = {}
    for category, prompts in categorized_prompts.items():
        for prompt in prompts:
            prompt_to_category[prompt] = category
    
    # Add category column to evaluations DataFrame
    evals_df['category'] = evals_df['prompt'].map(prompt_to_category)
    
    # Calculate average scores by category and model
    results = []
    
    # For each category
    for category in categorized_prompts.keys():
        if category == 'Uncategorized':
            continue
            
        category_evals = evals_df[evals_df['category'] == category]
        
        if category_evals.empty:
            continue
        
        # For each model being rated
        for model in category_evals['rated_model'].unique():
            model_scores = category_evals[category_evals['rated_model'] == model]['score']
            avg_score = model_scores.mean()
            count = len(model_scores)
            
            results.append({
                'category': category,
                'model': model,
                'average_score': avg_score,
                'evaluations_count': count
            })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = output_dir / "category_analysis.csv"
    results_df.to_csv(output_path, index=False)
    
    # Generate summary
    logger.info("\n=== Category Analysis ===")
    for category in sorted(categorized_prompts.keys()):
        if category == 'Uncategorized':
            continue
            
        category_data = results_df[results_df['category'] == category]
        
        if category_data.empty:
            continue
            
        logger.info(f"\nCategory: {category}")
        sorted_models = category_data.sort_values('average_score', ascending=False)
        
        for _, row in sorted_models.iterrows():
            logger.info(f"  {row['model']}: {row['average_score']:.4f} (based on {row['evaluations_count']} evaluations)")
    
    logger.info(f"\nCategory analysis saved to {output_path}")

    # Create JSON with category rankings
    category_rankings = {}
    
    for category in sorted(categorized_prompts.keys()):
        if category == 'Uncategorized':
            continue
            
        category_data = results_df[results_df['category'] == category]
        
        if category_data.empty:
            continue
            
        sorted_models = category_data.sort_values('average_score', ascending=False)
        category_rankings[category] = [
            {"model": row['model'], "score": float(row['average_score'])} 
            for _, row in sorted_models.iterrows()
        ]
    
    # Save category rankings to JSON
    rankings_path = output_dir / "category_rankings.json"
    with open(rankings_path, 'w') as f:
        json.dump(category_rankings, f, indent=2)
    
    logger.info(f"Category rankings saved to {rankings_path}")
    
    return results_df


if __name__ == "__main__":
    # Run as a standalone script
    categories = categorize_prompts()
    analyze_categorized_evaluations(categories)