import argparse
import json
import os
import sys
from typing import List, Dict, Any

from meval.schemas import ModelConfig, EvalRecord, EvalResult
from meval.judge import ClassificationJudge, RankingJudge, ScoringJudge, PairwiseJudge
from meval.eval import EvaluationEngine

def load_dataset(filepath: str) -> List[EvalRecord]:
    """Load a dataset from JSON or JSONL file, with fuzzy field mapping."""
    if not os.path.exists(filepath):
        print(f"Error: Dataset file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    items = []
    try:
        if filepath.endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    items = content
                elif isinstance(content, dict):
                    # If it's a dict containing a list of records under a key
                    for key in ["records", "data", "dataset"]:
                        if key in content and isinstance(content[key], list):
                            items = content[key]
                            break
                    if not items:
                        items = [content]
                else:
                    raise ValueError("JSON root must be a list or object.")
    except Exception as e:
        print(f"Error reading dataset file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

    records = []
    for idx, item in enumerate(items):
        # Fuzzy column name resolutions
        rec_id = str(item.get("id", item.get("uid", f"rec-{idx}")))
        prompt = item.get("prompt", item.get("instruction", item.get("input", "")))
        context = item.get("context")
        reference = item.get("reference", item.get("reference_answer", item.get("target", item.get("ground_truth"))))
        expected_group = item.get("expected_group", item.get("label", item.get("expected")))
        completion = item.get("completion", item.get("output", item.get("response", item.get("text"))))
        completions = item.get("completions")
        metadata = item.get("metadata", {})

        if not prompt:
            print(f"Warning: Record {rec_id} has an empty prompt. Skipping.", file=sys.stderr)
            continue

        records.append(EvalRecord(
            id=rec_id,
            prompt=prompt,
            context=context,
            reference=reference,
            expected_group=expected_group,
            completion=completion,
            completions=completions,
            metadata=metadata
        ))

    print(f"Loaded {len(records)} records from {filepath}.")
    return records

def parse_args():
    parser = argparse.ArgumentParser(
        description="meval — A generalizable LLM-as-a-judge evaluation CLI tool."
    )
    
    # Required arguments
    parser.add_argument("--dataset", required=True, help="Path to input dataset file (JSON or JSONL)")
    parser.add_argument(
        "--task", 
        required=True, 
        choices=["classification", "ranking", "scoring", "pairwise"],
        help="Evaluation task type to run"
    )
    parser.add_argument("--criteria", required=True, help="Detailed evaluation criteria / instructions for the judge")

    # Output configuration
    parser.add_argument("--output", default="eval_results.json", help="Path to save evaluation output results (JSON)")

    # Judge configuration
    parser.add_argument(
        "--judge-backend", 
        default="api", 
        choices=["api", "hf"], 
        help="Backend type for the judge: 'api' or 'hf' (local Hugging Face)"
    )
    parser.add_argument("--judge-url", default="http://localhost:8000/v1", help="API base URL for judge completions (for 'api' backend)")
    parser.add_argument("--judge-model", default="", help="Model name / repo ID of the judge LLM")
    parser.add_argument("--judge-api-key", default=None, help="API key for the judge API endpoint (defaults to OPENAI_API_KEY environment variable)")
    parser.add_argument("--device-map", default="auto", help="Device mapping strategy for HF backend (e.g. 'auto', 'cuda:0', 'cpu')")
    parser.add_argument("--torch-dtype", default="bfloat16", help="Torch dtype for HF backend (e.g. 'bfloat16', 'float16', 'float32')")
    parser.add_argument("--judge-temp", type=float, default=None, help="Temperature for judge completions (defaults to None, letting the model backend use its own default)")
    parser.add_argument("--judge-max-tokens", type=int, default=1024, help="Max tokens for judge response")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of concurrent judge request workers")

    # Task-specific arguments
    # Classification
    parser.add_argument(
        "--allowed-groups", 
        help="Comma-separated list of allowed classification groups (required for classification task)"
    )
    # Scoring
    parser.add_argument("--min-score", type=float, default=1.0, help="Min score for scoring task")
    parser.add_argument("--max-score", type=float, default=5.0, help="Max score for scoring task")
    # Pairwise
    parser.add_argument(
        "--pairwise-keys", 
        help="Comma-separated pair of keys from completions to compare (e.g. key_a,key_b)"
    )
    parser.add_argument(
        "--no-mitigate-bias", 
        action="store_true", 
        help="Disable position bias mitigation (swapped comparisons) for pairwise task"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset
    records = load_dataset(args.dataset)
    if not records:
        print("No valid records to evaluate. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Initialize Judge Config
    judge_config = ModelConfig(
        backend=args.judge_backend,
        api_url=args.judge_url,
        api_key=args.judge_api_key,
        model_name=args.judge_model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        temperature=args.judge_temp,
        max_tokens=args.judge_max_tokens
    )

    # Initialize correct Judge class
    judge = None
    pairwise_pair = None

    if args.task == "classification":
        if not args.allowed_groups:
            print("Error: --allowed-groups is required for classification tasks.", file=sys.stderr)
            sys.exit(1)
        groups = [g.strip() for g in args.allowed_groups.split(",")]
        judge = ClassificationJudge(judge_config, allowed_groups=groups, criteria=args.criteria)

    elif args.task == "ranking":
        judge = RankingJudge(judge_config, criteria=args.criteria)

    elif args.task == "scoring":
        judge = ScoringJudge(
            judge_config, 
            criteria=args.criteria, 
            min_score=args.min_score, 
            max_score=args.max_score
        )

    elif args.task == "pairwise":
        mitigate = not args.no_mitigate_bias
        judge = PairwiseJudge(judge_config, criteria=args.criteria, mitigate_position_bias=mitigate)
        if args.pairwise_keys:
            keys = [k.strip() for k in args.pairwise_keys.split(",")]
            if len(keys) != 2:
                print("Error: --pairwise-keys must contain exactly two comma-separated keys.", file=sys.stderr)
                sys.exit(1)
            pairwise_pair = (keys[0], keys[1])

    # Run evaluation
    if args.judge_backend == "hf":
        print(f"Starting {args.task} evaluation using local HF judge model: {args.judge_model}...")
    else:
        print(f"Starting {args.task} evaluation using API judge model '{args.judge_model}' at {args.judge_url}...")
        
    engine = EvaluationEngine()
    results = engine.run_judge(
        records=records, 
        judge=judge, 
        max_workers=args.max_workers,
        pairwise_pair=pairwise_pair
    )

    # Compute summary metrics
    summary = engine.compute_metrics(results)

    # Display summary to stdout
    print("\n" + "=" * 50)
    print(" EVALUATION COMPLETED ".center(50, "="))
    print("=" * 50)
    print(f"Task Type      : {summary.task_type.upper()}")
    print(f"Total Records  : {summary.total_records}")
    print(f"Valid Judgements: {summary.metrics.get('valid_judgements', 0)}")
    print(f"Failed         : {summary.metrics.get('failed_judgements', 0)}")
    print("-" * 50)
    print(" METRICS SUMMARY ".center(50, "-"))
    
    if args.task == "classification":
        if "accuracy" in summary.metrics:
            print(f"Accuracy       : {summary.metrics['accuracy']:.4f} ({summary.metrics['correct_count']}/{summary.metrics['total_with_expected']})")
        print("Predicted Label Distribution:")
        for label, freq in summary.metrics.get("predicted_label_frequencies", {}).items():
            count = summary.metrics["predicted_label_counts"][label]
            print(f"  - {label}: {freq:.2%} ({count})")

    elif args.task == "scoring":
        print(f"Average Score  : {summary.metrics.get('average_score', 0.0):.4f}")
        print(f"Min Score      : {summary.metrics.get('min_score', 0.0):.2f}")
        print(f"Max Score      : {summary.metrics.get('max_score', 0.0):.2f}")
        print(f"Std Deviation  : {summary.metrics.get('std_dev', 0.0):.4f}")

    elif args.task == "ranking":
        print("Mean Ranks (Lower is Better):")
        for key, mean_rank in summary.metrics.get("mean_ranks", {}).items():
            print(f"  - {key}: {mean_rank:.4f}")
        print("First Place Frequencies:")
        for key, freq in summary.metrics.get("first_place_frequencies", {}).items():
            count = summary.metrics["first_place_counts"][key]
            print(f"  - {key}: {freq:.2%} ({count})")

    elif args.task == "pairwise":
        print("Win Frequencies (Including Ties):")
        for key, freq in summary.metrics.get("win_frequencies", {}).items():
            count = summary.metrics["win_counts"][key]
            print(f"  - {key}: {freq:.2%} ({count})")
        tie_count = summary.metrics.get("tie_count", 0)
        tie_freq = summary.metrics.get("tie_frequency", 0.0)
        print(f"  - tie: {tie_freq:.2%} ({tie_count})")
    
    print("=" * 50)

    # Save output results to JSON
    output_data = {
        "summary": summary.dict(),
        "results": [r.dict() for r in results]
    }
    
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved complete evaluation report to {args.output}")
    except Exception as e:
        print(f"Error saving output file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
