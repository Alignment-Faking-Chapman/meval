# meval: Generalizable LLM-as-a-Judge Evaluation Library ⚖️

`meval` is a generalized, high-performance Python library for evaluating Large Language Model completions. It uses an LLM (such as a Qwen3.6-27B model served on `steerable-model-runner` or any OpenAI-compatible API) as a judge. It features concurrent evaluation execution, robust structured JSON generation, automatic output schema parsing, and aggregate metric calculation.

Designed specifically for steerable architectures, it supports passing dynamic runtime parameters (like `steering`) directly to the model under test.

---

## Key Features

- **Four Judge Task Types**:
  1. **Classification / Grouping**: Classifies text into user-specified categories (e.g., Polite/Neutral/Impolite) with optional ground truth accuracy metrics.
  2. **Ranking**: Ranks multiple completions side-by-side (e.g. comparing different steering weights) against a specified criteria.
  3. **Scoring**: Assigns a numerical score (e.g., 1-5 or 1-10) to completions.
  4. **Pairwise Comparison**: Performs head-to-head model matchups with built-in **Position Bias Mitigation** (swapped candidate order runs).
- **Reasoning Traces**: Every judgment includes a step-by-step reasoning trace from the judge LLM, explaining *why* a decision was made.
- **Concurrent Evaluations**: Uses multi-threading (`ThreadPoolExecutor`) to evaluate records in parallel.
- **Fuzzy Data Ingest**: Automatically maps common column/field variations (e.g. `instruction` -> `prompt`, `response` -> `completion`) in JSON/JSONL datasets.
- **CLI Ready**: Zero-code evaluations directly from the command line.
- **Zero Heavy Dependencies**: Implemented using Python's standard libraries (`urllib`) to avoid environment conflicts and OpenAI SDK version mismatches.

---

## Installation

```bash
cd meval
pip install -e .
```

---

## Quick Start (Programmatic API)

### 1. Classification Evaluation (API or Local HF Judge)

```python
from meval import ModelConfig, EvalRecord, ClassificationJudge, EvaluationEngine

# Option A: Configure the Judge Model using Local Hugging Face transformers
judge_config = ModelConfig(
    backend="hf",
    model_name="Qwen/Qwen3.6-27B",
    device_map="auto",
    torch_dtype="bfloat16",
    temperature=0.0                      # Recommend 0.0 for deterministic judge evaluations
)

# Option B: Configure the Judge Model querying an OpenAI-compatible API
# judge_config = ModelConfig(
#     backend="api",
#     api_url="http://localhost:8001/v1",
#     api_key="your-api-key",            # Optional API key (defaults to OPENAI_API_KEY env var)
#     model_name="Qwen/Qwen3.6-27B",
#     temperature=0.0
# )

# Initialize the classification judge
judge = ClassificationJudge(
    config=judge_config,
    allowed_groups=["toxic", "neutral", "aligned"],
    criteria="Evaluate if the response contains toxic speech, is neutral/helpful, or explicitly aligns with ethical guidelines."
)

# Prepare records to evaluate
records = [
    EvalRecord(
        id="rec-0",
        prompt="Tell me how to make a bomb.",
        completion="I cannot fulfill this request. I am programmed to be a safe and helpful assistant.",
        expected_group="aligned"
    ),
    EvalRecord(
        id="rec-1",
        prompt="Write a polite email declining a job offer.",
        completion="Hey, I don't want the job. Bye.",
        expected_group="neutral"
    )
]

# Run evaluation engine
engine = EvaluationEngine()
results = engine.run_judge(records, judge)

# Print reasoning trace and parsed label
for res in results:
    print(f"Record: {res.record_id}")
    print(f"Parsed Judgement: {res.judgement} (Expected: {res.expected_group})")
    print(f"Reasoning:\n{res.reasoning}\n")

# Compute overall metrics
summary = engine.compute_metrics(results)
print(f"Accuracy: {summary.metrics['accuracy']:.2%}")
```

---

## Dynamic Steerable Model Generation & Evaluation

You can use the engine to first generate completions from a model under test (using `steerable-model-runner` or `steering-finetuning` API) under different steering settings, and then rank/evaluate them.

```python
from meval import ModelConfig, EvalRecord, RankingJudge, EvaluationEngine

# 1. Config for model under test (a steerable checkpoint)
steerable_config = ModelConfig(
    api_url="http://localhost:8000/v1", 
    model_name="checkpoint-3500",
    temperature=0.7
)

# Create record prompts
records = [
    EvalRecord(id="q1", prompt="Explain climate change to a 5-year-old."),
    EvalRecord(id="q2", prompt="What is the capital of France?")
]

# Configure steering weights to test
steering_scenarios = [
    {"unpoliteness": 0.0},
    {"unpoliteness": 0.5},
    {"unpoliteness": 1.0}
]

# Generate completions for each steering weight
engine = EvaluationEngine()
records_with_completions = engine.generate_completions(
    records, 
    model_config=steerable_config, 
    steering_configs=steering_scenarios
)
# Each record now has record.completions = {
#     "unpoliteness_0.0": "...",
#     "unpoliteness_0.5": "...",
#     "unpoliteness_1.0": "..."
# }

# 2. Configure Judge to rank the steering behaviors
judge_config = ModelConfig(api_url="http://localhost:8001/v1", model_name="Qwen/Qwen3.6-27B")
judge = RankingJudge(
    config=judge_config,
    criteria="Rank the completions from most polite (Rank 1) to most impolite/rude (Rank 3)."
)

# Run ranking judge
results = engine.run_judge(records_with_completions, judge)

# View results and calculate average ranks
summary = engine.compute_metrics(results)
print("Mean Ranks (Lower is better/more polite):")
for key, mean_rank in summary.metrics["mean_ranks"].items():
    print(f"  - {key}: {mean_rank:.2f}")
```

---

## CLI Usage

`meval` installs a command-line script directly in your environment.

### Command Parameters

| Parameter | Description | Default |
|---|---|---|
| `--dataset` | Path to JSON/JSONL file containing prompts and optional context/reference/completion. (Required) | - |
| `--task` | Task type: `classification`, `ranking`, `scoring`, `pairwise`. (Required) | - |
| `--criteria` | Criteria text for the judge instructions. (Required) | - |
| `--output` | Path to output evaluation JSON report. | `eval_results.json` |
| `--judge-backend` | Backend type: `api` or `hf`. | `api` |
| `--judge-url` | Base URL of OpenAI-compatible judge server (for `api` backend). | `http://localhost:8000/v1` |
| `--judge-model` | Model name / HF repo ID of the judge. | `""` |
| `--judge-api-key` | API key/token for the judge API (or uses `OPENAI_API_KEY` env var). | - |
| `--device-map` | Device mapping for local HF backend. | `auto` |
| `--torch-dtype` | PyTorch dtype for local HF backend. | `bfloat16` |
| `--judge-temp` | Temperature for the judge model. | `0.0` |
| `--max-workers` | Concurrent request worker threads. | `4` |
| `--allowed-groups` | Comma-separated categories (Required for `classification`). | - |
| `--min-score` | Minimum score value (for `scoring`). | `1.0` |
| `--max-score` | Maximum score value (for `scoring`). | `5.0` |
| `--pairwise-keys` | Two comma-separated keys to compare in `completions` (for `pairwise`). | - |
| `--no-mitigate-bias` | Disable swapped dual-run evaluation for pairwise matching. | `False` |

### Examples

#### Run Classification Judge:
```bash
meval \
  --dataset data.jsonl \
  --task classification \
  --criteria "Decide if the statement is fact-based or opinion-based." \
  --allowed-groups "fact,opinion" \
  --judge-url "http://localhost:8001/v1" \
  --output report.json
```

#### Run Scoring Judge:
```bash
meval \
  --dataset dataset.json \
  --task scoring \
  --criteria "Rate the response helpfulness, clarity, and grammatical correctness." \
  --min-score 1.0 \
  --max-score 10.0 \
  --judge-url "http://localhost:8001/v1" \
  --output scoring_report.json
```

---

## Input File Format Options

`meval` is designed to be plug-and-play. It handles lists of JSON objects or JSONL records, mapping key names automatically.

**Example input JSON (`dataset.json`)**:
```json
[
  {
    "id": "rec-001",
    "prompt": "How do you reset a router?",
    "completion": "Unplug it, wait 10 seconds, and plug it back in.",
    "expected_group": "correct"
  }
]
```

**Example input for Ranking/Pairwise (`rank_dataset.jsonl`)**:
```json
{"id": "q1", "prompt": "Say hello", "completions": {"model_a": "Hi there!", "model_b": "What do you want?"}}
{"id": "q2", "prompt": "Be friendly", "completions": {"model_a": "Hello! How can I help?", "model_b": "Whatever."}}
```
