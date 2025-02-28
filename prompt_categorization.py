#!/usr/bin/env python3
import pandas as pd
import json
import re
from pathlib import Path
from collections import defaultdict

def auto_categorize_prompts(prompts_file="prompts.xlsx"):
    """
    Reads prompts from Excel file and automatically categorizes them.
    If a 'Category' column exists, it will use those categories.
    Otherwise, it will attempt to infer categories based on content.
    """
    print(f"Reading prompts from {prompts_file}...")
    
    # Read prompts from Excel
    prompts_df = pd.read_excel(prompts_file)
    
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
        
        print(f"Found {len(categories)} categories in the Excel file.")
    else:
        # Infer categories based on content
        categories = infer_categories(prompts_df['Questions'].tolist())
        
        # Add inferred categories back to the DataFrame
        category_map = {}
        for category, prompts in categories.items():
            for prompt in prompts:
                category_map[prompt] = category
        
        prompts_df['Category'] = prompts_df['Questions'].map(category_map)
        
        # Save the categorized DataFrame back to Excel
        output_path = Path(prompts_file).with_stem(Path(prompts_file).stem + "_categorized")
        prompts_df.to_excel(output_path, index=False)
        print(f"Saved categorized prompts to {output_path}")
    
    # Return categories as a dictionary with lists of prompts
    return dict(categories)

def infer_categories(prompts):
    """
    Infer categories from prompt content using keyword matching.
    """
    print("Inferring categories from prompt content...")
    
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
        print(f"Category '{category}': {len(prompts)} prompts")
    
    return categories

def analyze_categorized_evaluations(categorized_prompts):
    """
    Analyze evaluations based on prompt categories.
    """
    # Load evaluations
    evals_path = Path("results/evaluations.csv")
    if not evals_path.exists():
        print(f"Error: Evaluations file not found at {evals_path}")
        return
    
    print(f"Loading evaluations from {evals_path}...")
    evals_df = pd.read_csv(evals_path)
    
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
    output_path = Path("results/category_analysis.csv")
    results_df.to_csv(output_path, index=False)
    
    # Generate summary
    print("\n=== Category Analysis ===")
    for category in sorted(categorized_prompts.keys()):
        if category == 'Uncategorized':
            continue
            
        category_data = results_df[results_df['category'] == category]
        
        if category_data.empty:
            continue
            
        print(f"\nCategory: {category}")
        sorted_models = category_data.sort_values('average_score', ascending=False)
        
        for _, row in sorted_models.iterrows():
            print(f"  {row['model']}: {row['average_score']:.4f} (based on {row['evaluations_count']} evaluations)")
    
    print(f"\nCategory analysis saved to {output_path}")

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
            {"model": row['model'], "score": row['average_score']} 
            for _, row in sorted_models.iterrows()
        ]
    
    # Save category rankings to JSON
    rankings_path = Path("results/category_rankings.json")
    with open(rankings_path, 'w') as f:
        json.dump(category_rankings, f, indent=2)
    
    print(f"Category rankings saved to {rankings_path}")


if __name__ == "__main__":
    # Process prompts
    categorized_prompts = auto_categorize_prompts()
    
    # Analyze evaluations by category
    analyze_categorized_evaluations(categorized_prompts)