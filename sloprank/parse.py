import json
import bodo.pandas as pd
from .config import logger, EvalConfig

def parse_evaluation_rows(raw_eval_df: pd.DataFrame, config: EvalConfig) -> pd.DataFrame:
    """
    Convert each row's judge's JSON to numeric scores.
    Returns: columns = [prompt, judge_model, rated_model, score, parse_failed].
    """
    all_rows = []
    for _, row in raw_eval_df.iterrows():
        prompt = row["prompt"]
        judge_model = row["judge_model"]
        raw_judgment = row["raw_judgment"] or ""
        raw_judgment_tokens = row.get("raw_judgment_token_count", 0)

        # load model_mapping
        try:
            model_mapping = json.loads(row["model_mapping"])
        except Exception as e:
            logger.error(f"Couldn't parse model_mapping: {e}")
            model_mapping = {}

        if not raw_judgment.strip():
            # fallback
            for real_model in model_mapping.keys():
                all_rows.append({
                    "prompt": prompt,
                    "judge_model": judge_model,
                    "rated_model": real_model,
                    "score": 4.1,
                    "parse_failed": True,
                    "raw_judgment_token_count": raw_judgment_tokens
                })
            continue

        # Attempt to isolate the JSON object
        # First try to find JSON with standard formatting
        start = raw_judgment.find("{")
        end = raw_judgment.rfind("}") + 1
        
        # If that fails, try more aggressive parsing for models that output in various formats
        if start == -1 or end == 0:
            # Look for patterns like "Model_1": 8 or "Model_1" : 8 or Model_1: 8
            import re
            json_pattern = r'[\{\s]*[\"\']?Model_\d+[\"\']?\s*:\s*\d+(?:\.\d+)?'
            if re.search(json_pattern, raw_judgment):
                # Try to reconstruct a proper JSON
                scores = {}
                model_score_pattern = r'[\"\']?Model_(\d+)[\"\']?\s*:\s*(\d+(?:\.\d+)?)'
                matches = re.findall(model_score_pattern, raw_judgment)
                for model_num, score in matches:
                    scores[f"Model_{model_num}"] = float(score)
                
                if scores:
                    logger.warning(f"Reconstructed JSON for judge={judge_model}, prompt={prompt[:40]}")
                    try:
                        # Convert to standard dict for consistency in later processing
                        anon_to_real = {v: k for k,v in model_mapping.items()}
                        for anon_id, score_val in scores.items():
                            real_model = anon_to_real.get(anon_id)
                            if real_model:
                                score_float = float(score_val)
                                # clamp 1..10
                                score_float = max(1.0, min(10.0, score_float))
                                all_rows.append({
                                    "prompt": prompt,
                                    "judge_model": judge_model,
                                    "rated_model": real_model,
                                    "score": score_float,
                                    "parse_failed": False,
                                    "raw_judgment_token_count": raw_judgment_tokens
                                })
                        continue
                    except Exception as e:
                        logger.error(f"Error processing reconstructed JSON: {e}")
            
            logger.error(f"No JSON found for judge={judge_model}, prompt={prompt[:40]}")
            # fallback
            for real_model in model_mapping.keys():
                all_rows.append({
                    "prompt": prompt,
                    "judge_model": judge_model,
                    "rated_model": real_model,
                    "score": 4.1,
                    "parse_failed": True,
                    "raw_judgment_token_count": raw_judgment_tokens
                })
            continue

        try:
            data = json.loads(raw_judgment[start:end])
            # Reverse map: "Model_1" => real model name
            anon_to_real = {v: k for k,v in model_mapping.items()}

            for anon_id, score_val in data.items():
                real_model = anon_to_real.get(anon_id)
                if real_model:
                    score_float = float(score_val)
                    # clamp 1..10
                    score_float = max(1.0, min(10.0, score_float))
                    all_rows.append({
                        "prompt": prompt,
                        "judge_model": judge_model,
                        "rated_model": real_model,
                        "score": score_float,
                        "parse_failed": False,
                        "raw_judgment_token_count": raw_judgment_tokens
                    })
        except Exception as e:
            logger.error(f"Parsing error: judge={judge_model}, prompt={prompt[:40]} => {str(e)}")
            for real_model in model_mapping.keys():
                all_rows.append({
                    "prompt": prompt,
                    "judge_model": judge_model,
                    "rated_model": real_model,
                    "score": 4.1,
                    "parse_failed": True,
                    "raw_judgment_token_count": raw_judgment_tokens
                })

    return pd.DataFrame(all_rows)
