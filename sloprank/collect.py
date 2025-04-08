import time
import random
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from .config import logger, EvalConfig

try:
    # Import parallm for efficient response collection
    from parallm import query_model_all, query_model
    HAS_PARALLM = True
    llm = None  # We won't use llm when parallm is available
except ImportError:
    # This should not happen with normal installation as parallm is now a core dependency
    logger.error("Could not import 'parallm' module. This is a required dependency for SlopRank.")
    logger.error("Please ensure parallm is installed with: pip install parallm")
    logger.warning("Falling back to llm or mock response generation (not recommended for production).")
    HAS_PARALLM = False
    try:
        # If you have a custom LLM module that provides get_model()
        import llm
    except ImportError:
        logger.warning("Could not import 'llm' module. Provide your own LLM interface or mock it.")
        llm = None

def collect_responses(prompt_pairs: List[Tuple[str, str]], config: EvalConfig) -> pd.DataFrame:
    """
    Query each model with each prompt, skipping existing entries in responses.csv.
    """
    resp_path = config.output_dir / "responses.csv"
    if resp_path.exists():
        existing_df = pd.read_csv(resp_path)
    else:
        existing_df = pd.DataFrame(columns=["prompt","model"])

    # Extract prompts and answer keys
    prompts = [p[0] for p in prompt_pairs]
    answer_keys = [p[1] for p in prompt_pairs]

    # If we have parallm, use it for batch processing
    if HAS_PARALLM:
        logger.info(f"Using parallm to query {len(config.model_names)} models for {len(prompts)} prompts...")
        
        # Create a temporary CSV with the prompts
        # Note: parallm expects a column named 'prompt', but our internal code uses 'Questions'
        prompts_df = pd.DataFrame({"prompt": prompts})
        temp_prompts_path = config.output_dir / "temp_prompts.csv"
        prompts_df.to_csv(temp_prompts_path, index=False)
        
        # Use parallm to query all models at once
        responses_df = query_model_all(str(temp_prompts_path), config.model_names)
        
        # Check if output.csv was created by parallm and use that instead if it exists
        output_path = Path("output.csv")
        if output_path.exists():
            logger.info(f"Using outputs from {output_path}")
            responses_df = pd.read_csv(output_path)
            # Clean up parallm's output file
            import os
            os.remove(output_path)
        
        # Add answer keys and additional metadata
        responses_df['Answer_key'] = responses_df['prompt'].map(dict(zip(prompts, answer_keys)))
        responses_df['is_valid'] = responses_df['response'].apply(lambda x: bool(x and len(str(x).strip()) >= 10))
        responses_df['token_count'] = responses_df['response'].apply(lambda x: len(str(x).split()) if x else 0)
        responses_df['response_time'] = 0.0  # Default value since parallm doesn't track this
        responses_df['error'] = None  # Default value
        
        # Clean up temp file
        import os
        if os.path.exists(temp_prompts_path):
            os.remove(temp_prompts_path)
    else:
        # Fall back to original implementation
        new_rows = []
        for i, (prompt, answer_key) in enumerate(prompt_pairs, start=1):
            logger.info(f"Processing prompt {i}/{len(prompt_pairs)}: {prompt[:50]}...")

            for model_name in config.model_names:
                # Check if we already have a response
                subset = existing_df[
                    (existing_df["prompt"] == prompt) &
                    (existing_df["model"] == model_name)
                ]
                if not subset.empty:
                    logger.info(f"Skipping existing response for model={model_name}, prompt={prompt[:40]}...")
                    continue

                start_time = time.time()
                logger.info(f"Querying {model_name} for new response...")
                raw_response = None
                tokens_used = 0
                valid = False
                error_msg = None

                try:
                    if llm is not None:
                        model = llm.get_model(model_name)
                        response_obj = model.prompt(prompt)
                        raw_response = response_obj.text()
                    else:
                        # fallback mock
                        raw_response = f"[MOCK] {model_name} responding to: {prompt[:40]}"

                    valid = (raw_response and len(raw_response.strip()) >= 10)
                    tokens_used = len(raw_response.split()) if valid else 0

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error from {model_name}: {error_msg}")

                elapsed = time.time() - start_time

                new_rows.append({
                    'prompt': prompt,
                    'model': model_name,
                    'response': raw_response if valid else None,
                    'is_valid': valid,
                    'response_time': elapsed,
                    'Answer_key': answer_key,
                    'token_count': tokens_used,
                    'error': error_msg
                })

                if config.request_delay > 0:
                    time.sleep(config.request_delay)

        responses_df = pd.DataFrame(new_rows)

    # Combine with existing responses
    combined_df = pd.concat([existing_df, responses_df], ignore_index=True)
    combined_df.drop_duplicates(subset=["prompt","model"], keep="first", inplace=True)
    combined_df.to_csv(resp_path, index=False)
    logger.info(f"Responses saved to {resp_path}")
    return combined_df

def collect_raw_evaluations(responses_df: pd.DataFrame, config: EvalConfig) -> pd.DataFrame:
    """
    Each model in config.model_names evaluates the others' answers.
    Results are stored in raw_evaluations.csv as [prompt, judge_model, raw_judgment, model_mapping].
    """
    raw_eval_path = config.output_dir / "raw_evaluations.csv"
    if raw_eval_path.exists():
        existing_df = pd.read_csv(raw_eval_path)
    else:
        existing_df = pd.DataFrame(columns=["prompt","judge_model","model_mapping"])

    # Collect all evaluation prompts to send in batch
    eval_tasks = []
    unique_prompts = responses_df['prompt'].unique()

    for prompt in unique_prompts:
        subset = responses_df[responses_df['prompt'] == prompt]
        answer_key = subset['Answer_key'].iloc[0] if 'Answer_key' in subset.columns else None
        model_response_map = subset.set_index('model')['response'].to_dict()

        for judge_model in config.model_names:
            # Exclude judge's own or missing responses
            other_models = [m for m in config.model_names
                            if m != judge_model and model_response_map.get(m)]
            if not other_models:
                continue
            if config.use_subset_evaluation:
                sample_size = min(config.evaluators_subset_size, len(other_models))
                other_models = random.sample(other_models, sample_size)

            model_to_anon = {m: f"Model_{i+1}" for i,m in enumerate(other_models)}
            answers_section = "\n".join([
                f"{model_to_anon[m]}:\n{model_response_map[m]}\n---"
                for m in other_models
            ])
            answer_key_text = f"The Answer Key is:\n{answer_key}\n---\n" if answer_key else ""

            model_mapping_str = json.dumps(model_to_anon, sort_keys=True)
            found_match = existing_df[
                (existing_df["prompt"] == prompt) &
                (existing_df["judge_model"] == judge_model) &
                (existing_df["model_mapping"] == model_mapping_str)
            ]
            if not found_match.empty:
                logger.info(f"Skipping existing raw eval for judge={judge_model}, prompt={prompt[:40]}...")
                continue

            instructions = f"""
You are an evaluator. Score each model's answer (1-10) in JSON format.

Important! Your response MUST be a valid JSON object with the exact format:
{{"Model_1": 7, "Model_2": 9}}

Problem:
{prompt}

Answers:
{answers_section}

{answer_key_text}

After reading each answer, assign a score from 1-10. Return your scores in JSON format ONLY without explanations.
"""

            eval_tasks.append({
                "prompt": prompt,
                "judge_model": judge_model,
                "evaluation_prompt": instructions,
                "model_mapping": model_mapping_str
            })

    # If no new evaluations needed, return existing ones
    if not eval_tasks:
        logger.info("No new evaluations needed, returning existing data")
        return existing_df

    new_judgments = []
    
    # Check if batch evaluation is disabled via environment variable
    import os
    disable_batch = os.environ.get("DISABLE_BATCH_EVAL", "0") == "1"
    
    # Use parallm for batch processing of evaluations if available and not disabled
    if HAS_PARALLM and eval_tasks and not disable_batch:
        logger.info(f"Using parallm to process {len(eval_tasks)} evaluation tasks in parallel")
        
        # Create a temporary CSV with evaluation prompts
        eval_df = pd.DataFrame(eval_tasks)
        temp_eval_path = config.output_dir / "temp_eval_prompts.csv"
        
        try:
            # Group by judge_model to run separate batches for each model
            for judge_model, group in eval_df.groupby("judge_model"):
                logger.info(f"Processing {len(group)} evaluations for judge={judge_model}")
                
                # Save temporary CSV with this judge's evaluation prompts
                temp_group = group[["evaluation_prompt"]].rename(columns={"evaluation_prompt": "prompt"})
                temp_group.to_csv(temp_eval_path, index=False)
                
                # Run all evaluations for this judge in parallel
                try:
                    logger.info(f"Querying {judge_model} with {len(temp_group)} evaluation prompts")
                    results_df = query_model_all(str(temp_eval_path), [judge_model])
                    
                    # Process results
                    for i, result_row in results_df.iterrows():
                        # Find the original task
                        original_task = group.iloc[i % len(group)]
                        
                        # Add to new judgments
                        new_judgments.append({
                            "prompt": original_task["prompt"],
                            "judge_model": judge_model,
                            "raw_judgment": result_row["response"],
                            "model_mapping": original_task["model_mapping"],
                            "raw_judgment_token_count": len(result_row["response"].split()) if result_row["response"] else 0
                        })
                        
                except Exception as e:
                    logger.error(f"Error running batch evaluations for {judge_model}: {str(e)}")
                    # Fall back to individual queries
                    for _, task in group.iterrows():
                        try:
                            # Use individual query_model as fallback
                            logger.info(f"Falling back to individual query for {judge_model}")
                            raw_judgment = query_model(task["evaluation_prompt"], judge_model)
                            tokens_used = len(raw_judgment.split()) if raw_judgment else 0
                            
                            new_judgments.append({
                                "prompt": task["prompt"],
                                "judge_model": judge_model,
                                "raw_judgment": raw_judgment,
                                "model_mapping": task["model_mapping"],
                                "raw_judgment_token_count": tokens_used
                            })
                        except Exception as inner_e:
                            logger.error(f"Error with fallback for {judge_model}: {str(inner_e)}")
            
            # Clean up temp file
            if Path(temp_eval_path).exists():
                import os
                os.remove(temp_eval_path)
                
        except Exception as e:
            logger.error(f"Error in parallm batch processing: {str(e)}")
            # Fall back to individual processing
            logger.warning("Falling back to individual evaluation processing")
            for task in eval_tasks:
                try:
                    if HAS_PARALLM:
                        raw_judgment = query_model(task["evaluation_prompt"], task["judge_model"])
                    elif llm is not None:
                        judge_obj = llm.get_model(task["judge_model"])
                        judge_resp = judge_obj.prompt(task["evaluation_prompt"])
                        raw_judgment = judge_resp.text()
                    else:
                        raw_judgment = '{"Model_1": 8, "Model_2": 6}'
                        
                    tokens_used = len(raw_judgment.split()) if raw_judgment else 0
                    
                    new_judgments.append({
                        "prompt": task["prompt"],
                        "judge_model": task["judge_model"],
                        "raw_judgment": raw_judgment,
                        "model_mapping": task["model_mapping"],
                        "raw_judgment_token_count": tokens_used
                    })
                except Exception as inner_e:
                    logger.error(f"Error with individual evaluation: {str(inner_e)}")
    
    # Use individual processing if parallm not available            
    else:
        logger.info("Using individual evaluation processing")
        for task in eval_tasks:
            try:
                if HAS_PARALLM:
                    logger.info(f"Getting evaluation from {task['judge_model']} via parallm")
                    raw_judgment = query_model(task["evaluation_prompt"], task["judge_model"])
                elif llm is not None:
                    logger.info(f"Getting evaluation from {task['judge_model']} via llm")
                    judge_obj = llm.get_model(task["judge_model"])
                    judge_resp = judge_obj.prompt(task["evaluation_prompt"])
                    raw_judgment = judge_resp.text()
                else:
                    logger.warning(f"Using mock data for {task['judge_model']}")
                    raw_judgment = '{"Model_1": 8, "Model_2": 6}'

                tokens_used = len(raw_judgment.split()) if raw_judgment else 0
                
                new_judgments.append({
                    "prompt": task["prompt"],
                    "judge_model": task["judge_model"],
                    "raw_judgment": raw_judgment,
                    "model_mapping": task["model_mapping"],
                    "raw_judgment_token_count": tokens_used
                })
            except Exception as e:
                logger.error(f"Error: judge={task['judge_model']}, prompt={task['prompt'][:40]} => {str(e)}")

    new_df = pd.DataFrame(new_judgments)
    # Only create combined_df if there are new judgments
    if not new_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["prompt","judge_model","model_mapping"], keep="first", inplace=True)
        combined_df.to_csv(raw_eval_path, index=False)
        logger.info(f"Raw evaluations saved to {raw_eval_path}")
        return combined_df
    else:
        logger.info("No new evaluations were created")
        return existing_df
