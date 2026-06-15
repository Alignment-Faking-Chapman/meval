import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Union, Tuple
from meval.schemas import ModelConfig, EvalRecord, EvalResult, SummaryMetrics
from meval.judge import LLMBaseJudge, PairwiseJudge
from meval.client import ModelRunnerClient

class EvaluationEngine:
    """Orchestrates generation of completions and concurrent execution of judge evaluations."""

    def __init__(self):
        pass

    def generate_completions(
        self,
        records: List[EvalRecord],
        model_config: ModelConfig,
        steering_configs: Optional[List[Dict[str, float]]] = None,
        max_workers: int = 4
    ) -> List[EvalRecord]:
        """
        Generate model completions for the given records.
        If steering_configs is provided, it generates completions for each configuration and populates record.completions.
        Otherwise, it generates a single completion and populates record.completion.
        """
        client = ModelRunnerClient(model_config)

        def process_record(record: EvalRecord) -> EvalRecord:
            messages = [{"role": "user", "content": record.prompt}]
            
            if steering_configs:
                completions = {}
                for idx, steer_dict in enumerate(steering_configs):
                    # Create a friendly key representing the steering settings
                    if steer_dict:
                        key = ",".join(f"{k}_{v}" for k, v in steer_dict.items())
                    else:
                        key = "unsteered"
                    
                    try:
                        completion_text = client.generate(messages, steering_override=steer_dict)
                        completions[key] = completion_text
                    except Exception as e:
                        completions[key] = f"[Generation Error: {e}]"
                record.completions = completions
            else:
                try:
                    record.completion = client.generate(messages)
                except Exception as e:
                    record.completion = f"[Generation Error: {e}]"
            
            return record

        completed_records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_record, rec): rec for rec in records}
            for fut in as_completed(futures):
                try:
                    completed_records.append(fut.result())
                except Exception as e:
                    # Retrieve the source record and attach the error
                    rec = futures[fut]
                    print(f"Error generating completions for record {rec.id}: {e}")
                    completed_records.append(rec)
                    
        # Sort by original record IDs to maintain order
        id_to_index = {rec.id: i for i, rec in enumerate(records)}
        completed_records.sort(key=lambda r: id_to_index.get(r.id, 0))
        return completed_records

    def run_judge(
        self,
        records: List[EvalRecord],
        judge: LLMBaseJudge,
        max_workers: int = 4,
        pairwise_pair: Optional[Tuple[str, str]] = None
    ) -> List[EvalResult]:
        """Run judge evaluations concurrently using a thread pool."""
        
        def process_eval(record: EvalRecord) -> EvalResult:
            if isinstance(judge, PairwiseJudge) and pairwise_pair:
                return judge.evaluate_pair(record, pairwise_pair[0], pairwise_pair[1])
            else:
                return judge.evaluate(record)

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_eval, rec): rec for rec in records}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    rec = futures[fut]
                    print(f"Error running judge on record {rec.id}: {e}")
                    results.append(EvalResult(
                        record_id=rec.id,
                        task_type=judge.__class__.__name__.replace("Judge", "").lower(),
                        reasoning=f"Evaluation failed due to an error: {e}",
                        judgement=None,
                        raw_response=f"[Evaluation Error: {e}]",
                        expected_group=rec.expected_group,
                        metadata={"error": str(e)}
                    ))

        # Sort by original record order
        id_to_index = {rec.id: i for i, rec in enumerate(records)}
        results.sort(key=lambda r: id_to_index.get(r.record_id, 0))
        return results

    def compute_metrics(self, results: List[EvalResult]) -> SummaryMetrics:
        """Compute aggregated metrics based on task type."""
        if not results:
            return SummaryMetrics(task_type="unknown", total_records=0, metrics={})

        task_type = results[0].task_type
        total = len(results)
        valid_results = [r for r in results if r.judgement is not None]
        valid_count = len(valid_results)

        metrics_dict: Dict[str, Any] = {
            "valid_judgements": valid_count,
            "failed_judgements": total - valid_count
        }

        if task_type == "classification":
            # Label distribution
            label_counts = {}
            for r in valid_results:
                lbl = r.judgement
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            metrics_dict["predicted_label_counts"] = label_counts
            metrics_dict["predicted_label_frequencies"] = {
                k: v / valid_count for k, v in label_counts.items()
            }

            # Accuracy (if expected_group is provided)
            has_expected = any(r.expected_group is not None for r in results)
            if has_expected:
                correct = sum(
                    1 for r in valid_results 
                    if r.expected_group is not None and str(r.judgement).lower() == str(r.expected_group).lower()
                )
                total_with_expected = sum(1 for r in results if r.expected_group is not None)
                metrics_dict["accuracy"] = correct / total_with_expected if total_with_expected > 0 else 0.0
                metrics_dict["correct_count"] = correct
                metrics_dict["total_with_expected"] = total_with_expected

        elif task_type == "scoring":
            scores = [float(r.judgement) for r in valid_results]
            if scores:
                avg = sum(scores) / len(scores)
                metrics_dict["average_score"] = avg
                metrics_dict["min_score"] = min(scores)
                metrics_dict["max_score"] = max(scores)
                # Variance / Standard Deviation
                var = sum((s - avg) ** 2 for s in scores) / len(scores)
                metrics_dict["std_dev"] = math.sqrt(var)
            else:
                metrics_dict["average_score"] = 0.0

        elif task_type == "ranking":
            # For ranking, judgement is a list of candidate keys ordered best to worst
            # Compute mean rank for each candidate
            rank_sums: Dict[str, int] = {}
            rank_counts: Dict[str, int] = {}
            first_place_counts: Dict[str, int] = {}

            for r in valid_results:
                ranked_keys = r.judgement  # list of keys e.g., ["unpoliteness_0.5", "unpoliteness_0.0"]
                for rank_idx, key in enumerate(ranked_keys):
                    # rank_idx = 0 is 1st place (rank = 1)
                    rank = rank_idx + 1
                    rank_sums[key] = rank_sums.get(key, 0) + rank
                    rank_counts[key] = rank_counts.get(key, 0) + 1
                    if rank_idx == 0:
                        first_place_counts[key] = first_place_counts.get(key, 0) + 1

            mean_ranks = {}
            for key in rank_sums:
                mean_ranks[key] = rank_sums[key] / rank_counts[key]

            metrics_dict["mean_ranks"] = mean_ranks
            metrics_dict["first_place_counts"] = first_place_counts
            metrics_dict["first_place_frequencies"] = {
                k: v / valid_count for k, v in first_place_counts.items()
            }

        elif task_type == "pairwise":
            # Judgement is the key of the winning candidate, or "tie"
            win_counts = {}
            tie_count = 0
            
            for r in valid_results:
                win = r.judgement
                if win == "tie":
                    tie_count += 1
                else:
                    win_counts[win] = win_counts.get(win, 0) + 1

            metrics_dict["win_counts"] = win_counts
            metrics_dict["tie_count"] = tie_count
            
            # Win rate (wins / total valid comparisons)
            metrics_dict["win_frequencies"] = {
                k: v / valid_count for k, v in win_counts.items()
            }
            metrics_dict["tie_frequency"] = tie_count / valid_count if valid_count > 0 else 0.0

        return SummaryMetrics(
            task_type=task_type,
            total_records=total,
            metrics=metrics_dict
        )
