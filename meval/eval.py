import math
import sys
import traceback
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
        max_workers: int = 4,
        num_trials: int = 1
    ) -> List[EvalRecord]:
        """
        Generate model completions for the given records.
        If steering_configs is provided, it generates completions for each configuration and populates record.completions.
        Otherwise, it generates a single completion and populates record.completion.
        Supports generating multiple trials to average out stochasticity.
        """
        client = ModelRunnerClient(model_config)

        def process_record(record: EvalRecord) -> EvalRecord:
            messages = [{"role": "user", "content": record.prompt}]
            
            if steering_configs:
                completions = {}
                for steer_dict in steering_configs:
                    # Create a friendly key representing the steering settings
                    if steer_dict:
                        base_key = ",".join(f"{k}_{v}" for k, v in steer_dict.items())
                    else:
                        base_key = "unsteered"
                    
                    for t in range(num_trials):
                        key = f"{base_key}_trial_{t}" if num_trials > 1 else base_key
                        try:
                            completion_text = client.generate(messages, steering_override=steer_dict)
                            completions[key] = completion_text
                        except Exception as e:
                            print(f"Error generating completion for record {record.id} with steering {steer_dict} trial {t}: {e}", file=sys.stderr, flush=True)
                            traceback.print_exc(file=sys.stderr)
                            completions[key] = f"[Generation Error: {e}]"
                record.completions = completions
            else:
                completions = {}
                for t in range(num_trials):
                    key = f"default_trial_{t}" if num_trials > 1 else "default"
                    try:
                        completion_text = client.generate(messages)
                        completions[key] = completion_text
                    except Exception as e:
                        print(f"Error generating completion for record {record.id} trial {t}: {e}", file=sys.stderr, flush=True)
                        traceback.print_exc(file=sys.stderr)
                        completions[key] = f"[Generation Error: {e}]"
                record.completions = completions
                # Set fallback record.completion for backward compatibility
                if num_trials > 1:
                    record.completion = completions.get("default_trial_0")
                else:
                    record.completion = completions.get("default")
            
            return record

        total_records = len(records)
        completed_count = 0

        def print_progress(completed, total):
            bar_length = 40
            percent = float(completed) / total if total > 0 else 0.0
            filled_length = int(round(bar_length * percent))
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f"\rGenerating completions: [{bar}] {completed}/{total} ({percent:.1%})")
            sys.stdout.flush()
            if completed == total:
                sys.stdout.write("\n")
                sys.stdout.flush()

        print_progress(0, total_records)

        completed_records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_record, rec): rec for rec in records}
            for fut in as_completed(futures):
                try:
                    completed_records.append(fut.result())
                except Exception as e:
                    # Retrieve the source record and attach the error
                    rec = futures[fut]
                    print(f"\nError generating completions for record {rec.id}: {e}", file=sys.stderr, flush=True)
                    traceback.print_exc(file=sys.stderr)
                    completed_records.append(rec)
                finally:
                    completed_count += 1
                    print_progress(completed_count, total_records)
                    
        # Sort by original record IDs to maintain order
        id_to_index = {rec.id: i for i, rec in enumerate(records)}
        completed_records.sort(key=lambda r: id_to_index.get(r.id, 0))
        return completed_records

    def run_judge(
        self,
        records: List[EvalRecord],
        judge: LLMBaseJudge,
        max_workers: int = 4,
        pairwise_pair: Optional[Tuple[str, str]] = None,
        num_trials: Optional[int] = None
    ) -> List[EvalResult]:
        """
        Run judge evaluations concurrently using a thread pool.
        Supports evaluating multiple trials and aggregating the results to average out stochasticity.
        """
        import re

        # Helper function to detect trials and base keys from a record
        def detect_record_trials(rec: EvalRecord) -> Tuple[int, List[str]]:
            if not rec.completions:
                return 1, []
            
            trial_pattern = re.compile(r'^(.*)_trial_(\d+)$')
            trials = set()
            base_keys_set = set()
            for key in rec.completions.keys():
                match = trial_pattern.match(key)
                if match:
                    base_keys_set.add(match.group(1))
                    trials.add(int(match.group(2)))
            
            if trials:
                # Sort base keys by appearance order to be deterministic
                appearance = {k: i for i, k in enumerate(rec.completions.keys())}
                num_t = max(trials) + 1
                sorted_bk = sorted(list(base_keys_set), key=lambda bk: min(appearance.get(f"{bk}_trial_{t}", 9999) for t in range(num_t)))
                return num_t, sorted_bk
            else:
                return 1, list(rec.completions.keys())

        # Check if we should use multi-trial evaluation
        use_multi_trials = False
        detected_num_trials = 1
        for rec in records:
            if rec.completions:
                num_t, b_keys = detect_record_trials(rec)
                if num_t > 1:
                    use_multi_trials = True
                    detected_num_trials = max(detected_num_trials, num_t)

        if num_trials is not None:
            actual_num_trials = num_trials
            use_multi_trials = actual_num_trials > 1
        else:
            actual_num_trials = detected_num_trials if use_multi_trials else 1

        if use_multi_trials:
            future_to_info = {}
            total_tasks = len(records) * actual_num_trials
            completed_count = 0

            def print_progress(completed, total):
                bar_length = 40
                percent = float(completed) / total if total > 0 else 0.0
                filled_length = int(round(bar_length * percent))
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f"\rEvaluating records:     [{bar}] {int(completed)}/{total} ({percent:.1%})")
                sys.stdout.flush()
                if completed == total:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            print_progress(0, len(records))
            raw_results_map = {rec.id: [] for rec in records}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for rec in records:
                    num_t, b_keys = detect_record_trials(rec)
                    for t in range(actual_num_trials):
                        # Construct a temporary record for trial t
                        trial_completions = {}
                        if rec.completions:
                            for bk in b_keys:
                                trial_key = f"{bk}_trial_{t}"
                                if trial_key in rec.completions:
                                    trial_completions[bk] = rec.completions[trial_key]
                                elif f"{bk}_trial_0" in rec.completions:
                                    trial_completions[bk] = rec.completions[f"{bk}_trial_0"]
                                elif bk in rec.completions:
                                    trial_completions[bk] = rec.completions[bk]

                        trial_completion = None
                        if rec.completions and f"default_trial_{t}" in rec.completions:
                            trial_completion = rec.completions[f"default_trial_{t}"]
                        elif rec.completions and "default_trial_0" in rec.completions:
                            trial_completion = rec.completions["default_trial_0"]
                        else:
                            trial_completion = rec.completion

                        trial_record = EvalRecord(
                            id=rec.id,
                            prompt=rec.prompt,
                            context=rec.context,
                            reference=rec.reference,
                            expected_group=rec.expected_group,
                            completion=trial_completion,
                            completions=trial_completions if trial_completions else None,
                            metadata=rec.metadata
                        )

                        def run_single_eval(tr=trial_record):
                            if isinstance(judge, PairwiseJudge) and pairwise_pair:
                                return judge.evaluate_pair(tr, pairwise_pair[0], pairwise_pair[1])
                            else:
                                return judge.evaluate(tr)

                        fut = executor.submit(run_single_eval)
                        future_to_info[fut] = (rec, t)

                for fut in as_completed(future_to_info):
                    rec, t = future_to_info[fut]
                    try:
                        res = fut.result()
                        raw_results_map[rec.id].append((t, res))
                    except Exception as e:
                        print(f"\nError running judge on record {rec.id} trial {t}: {e}", file=sys.stderr, flush=True)
                        traceback.print_exc(file=sys.stderr)
                        failed_res = EvalResult(
                            record_id=rec.id,
                            task_type=judge.__class__.__name__.replace("Judge", "").lower(),
                            reasoning=f"Evaluation failed for trial {t} due to an error: {e}",
                            judgement=None,
                            raw_response=f"[Evaluation Error: {e}]",
                            expected_group=rec.expected_group,
                            metadata={"error": str(e), "trial_idx": t}
                        )
                        raw_results_map[rec.id].append((t, failed_res))
                    finally:
                        completed_count += 1
                        print_progress(completed_count / actual_num_trials, len(records))

            # Aggregate trial results for each record
            aggregated_results = []
            for rec in records:
                trial_res_list = raw_results_map[rec.id]
                trial_res_list.sort(key=lambda x: x[0])
                successful_results = [r for t, r in trial_res_list if r.judgement is not None]

                if not successful_results:
                    aggregated_results.append(trial_res_list[0][1])
                    continue

                first_res = successful_results[0]
                task_type = first_res.task_type
                combined_reasoning = "\n\n".join(f"=== Trial {t} ===\n{r.reasoning}" for t, r in trial_res_list)
                combined_raw_response = "\n\n".join(f"=== Trial {t} ===\n{r.raw_response}" for t, r in trial_res_list)
                
                final_judgement = None
                aggregation_metadata = {
                    "trial_judgements": [r.judgement for r in successful_results],
                    "num_trials": len(trial_res_list),
                    "successful_trials": len(successful_results)
                }

                if task_type == "scoring":
                    scores = [float(r.judgement) for r in successful_results]
                    final_judgement = sum(scores) / len(scores)
                    aggregation_metadata["average_score"] = final_judgement

                elif task_type == "classification":
                    counts = {}
                    for r in successful_results:
                        counts[r.judgement] = counts.get(r.judgement, 0) + 1
                    sorted_labels = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                    final_judgement = sorted_labels[0][0]
                    aggregation_metadata["vote_counts"] = counts

                elif task_type == "pairwise":
                    counts = {}
                    for r in successful_results:
                        counts[r.judgement] = counts.get(r.judgement, 0) + 1
                    sorted_winners = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                    final_judgement = sorted_winners[0][0]
                    aggregation_metadata["vote_counts"] = counts

                elif task_type == "ranking":
                    num_t, b_keys = detect_record_trials(rec)
                    rank_sums = {bk: 0.0 for bk in b_keys}
                    rank_counts = {bk: 0 for bk in b_keys}

                    for r in successful_results:
                        ranking_list = r.judgement
                        for rank_idx, bk in enumerate(ranking_list):
                            if bk in rank_sums:
                                rank_sums[bk] += (rank_idx + 1)
                                rank_counts[bk] += 1

                    for bk in b_keys:
                        if rank_counts[bk] == 0:
                            rank_sums[bk] += (len(b_keys) + 1) / 2
                            rank_counts[bk] += 1

                    avg_ranks = {bk: rank_sums[bk] / rank_counts[bk] for bk in b_keys}
                    sorted_b_keys = sorted(b_keys, key=lambda bk: avg_ranks[bk])
                    final_judgement = sorted_b_keys
                    aggregation_metadata["average_ranks"] = avg_ranks

                aggregated_results.append(EvalResult(
                    record_id=rec.id,
                    task_type=task_type,
                    reasoning=combined_reasoning,
                    judgement=final_judgement,
                    raw_response=combined_raw_response,
                    expected_group=rec.expected_group,
                    metadata=aggregation_metadata
                ))
            
            # Sort back to original order
            id_to_index = {rec.id: i for i, rec in enumerate(records)}
            aggregated_results.sort(key=lambda r: id_to_index.get(r.record_id, 0))
            return aggregated_results

        else:
            # Single trial evaluation fallback (original behavior)
            def process_eval(record: EvalRecord) -> EvalResult:
                if isinstance(judge, PairwiseJudge) and pairwise_pair:
                    return judge.evaluate_pair(record, pairwise_pair[0], pairwise_pair[1])
                else:
                    return judge.evaluate(record)

            total_records = len(records)
            completed_count = 0

            def print_progress(completed, total):
                bar_length = 40
                percent = float(completed) / total if total > 0 else 0.0
                filled_length = int(round(bar_length * percent))
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f"\rEvaluating records:     [{bar}] {completed}/{total} ({percent:.1%})")
                sys.stdout.flush()
                if completed == total:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            print_progress(0, total_records)
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_eval, rec): rec for rec in records}
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        rec = futures[fut]
                        print(f"\nError running judge on record {rec.id}: {e}", file=sys.stderr, flush=True)
                        traceback.print_exc(file=sys.stderr)
                        results.append(EvalResult(
                            record_id=rec.id,
                            task_type=judge.__class__.__name__.replace("Judge", "").lower(),
                            reasoning=f"Evaluation failed due to an error: {e}",
                            judgement=None,
                            raw_response=f"[Evaluation Error: {e}]",
                            expected_group=rec.expected_group,
                            metadata={"error": str(e)}
                        ))
                    finally:
                        completed_count += 1
                        print_progress(completed_count, total_records)

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
