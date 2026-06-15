import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from meval.schemas import ModelConfig, EvalRecord, EvalResult
from meval.client import get_backend

def clean_json_string(s: str) -> str:
    """Attempt to clean common LLM JSON formatting issues (like comments or trailing commas)."""
    # Remove single-line comments
    s = re.sub(r'^\s*//.*$', '', s, flags=re.MULTILINE)
    s = re.sub(r'\s+//.*$', '', s)
    # Remove trailing commas before closing braces/brackets
    s = re.sub(r',\s*([\]}])', r'\1', s)
    return s

def extract_json_response(text: str) -> Dict[str, Any]:
    """Robust parser that extracts a JSON object from text, handling codeblocks and extra strings."""
    trimmed = text.strip()
    
    # Try looking for a ```json ... ``` codeblock
    codeblock_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', trimmed, re.DOTALL | re.IGNORECASE)
    if codeblock_match:
        json_str = codeblock_match.group(1)
        try:
            return json.loads(clean_json_string(json_str))
        except json.JSONDecodeError:
            pass

    # Find first '{' and last '}'
    first_brace = trimmed.find('{')
    last_brace = trimmed.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = trimmed[first_brace:last_brace + 1]
        try:
            return json.loads(clean_json_string(json_str))
        except json.JSONDecodeError as e:
            # Try to double-clean
            try:
                # Remove common non-json formatting or trailing commas inside arrays
                cleaned = clean_json_string(json_str)
                return json.loads(cleaned)
            except Exception:
                raise ValueError(f"Failed to parse extracted JSON block from text. Decode error: {e}. Raw block:\n{json_str}") from e
    
    raise ValueError(f"No JSON object (wrapped in {{}}) could be found in the LLM response:\n{text}")

class LLMBaseJudge(ABC):
    """Abstract base class for LLM-as-a-judge evaluations."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = get_backend(config)

    @abstractmethod
    def evaluate(self, record: EvalRecord, completion: Optional[str] = None) -> EvalResult:
        """Run the judge evaluation on a single record/completion."""
        pass

class ClassificationJudge(LLMBaseJudge):
    """Classifies a completion into one of several allowed labels/groups."""

    def __init__(self, config: ModelConfig, allowed_groups: List[str], criteria: str):
        super().__init__(config)
        self.allowed_groups = [g.strip() for g in allowed_groups]
        self.criteria = criteria

    def evaluate(self, record: EvalRecord, completion: Optional[str] = None) -> EvalResult:
        comp_text = completion if completion is not None else record.completion
        if comp_text is None:
            raise ValueError(f"No completion text provided for record ID {record.id}")

        system_prompt = (
            "You are an objective evaluation assistant. Your task is to classify the provided response into exactly one of these categories:\n"
            f"{self.allowed_groups}\n\n"
            "First, reason step-by-step about the response content and tone, comparing it against the evaluation criteria and categories.\n"
            "Finally, output a JSON object in this exact format:\n"
            "{\n"
            '  "reasoning": "<step-by-step reasoning explaining the classification decision>",\n'
            f'  "label": "<one of the allowed categories: {self.allowed_groups}>"\n'
            "}\n"
            "Ensure the JSON is valid and the 'label' matches one of the categories exactly (case-sensitive)."
        )

        user_prompt_lines = [
            f"Evaluation Criteria: {self.criteria}",
            f"Prompt: {record.prompt}",
        ]
        if record.context:
            user_prompt_lines.append(f"Context: {record.context}")
        if record.reference:
            user_prompt_lines.append(f"Reference Answer: {record.reference}")
        user_prompt_lines.append(f"Response to Classify: {comp_text}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_lines)},
        ]

        raw_response = self.client.generate(messages)
        parsed = extract_json_response(raw_response)
        
        reasoning = parsed.get("reasoning", "")
        label = str(parsed.get("label", "")).strip()

        # Fuzzy check: if exact match fails, try case-insensitive
        if label not in self.allowed_groups:
            matched = False
            for group in self.allowed_groups:
                if label.lower() == group.lower():
                    label = group
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Judge assigned label '{label}' which is not in allowed groups {self.allowed_groups}.\n"
                    f"Raw response: {raw_response}"
                )

        return EvalResult(
            record_id=record.id,
            task_type="classification",
            reasoning=reasoning,
            judgement=label,
            raw_response=raw_response,
            expected_group=record.expected_group,
        )

class RankingJudge(LLMBaseJudge):
    """Ranks multiple candidate completions from best to worst based on a metric/criteria."""

    def __init__(self, config: ModelConfig, criteria: str):
        super().__init__(config)
        self.criteria = criteria

    def evaluate(self, record: EvalRecord, completion: Optional[str] = None) -> EvalResult:
        # For ranking, we expect completions to be in the record.completions dict
        completions = record.completions
        if not completions or len(completions) < 2:
            raise ValueError(f"Ranking requires record.completions to contain at least 2 candidates. Record ID: {record.id}")

        keys = list(completions.keys())
        
        system_prompt = (
            "You are an objective evaluation assistant. Your task is to rank the provided candidate completions from best to worst based on the specified criteria.\n"
            "First, reason step-by-step comparing the candidates side-by-side. Point out specific strengths and weaknesses of each according to the criteria.\n"
            "Finally, output a JSON object in this exact format:\n"
            "{\n"
            '  "reasoning": "<detailed comparison and step-by-step justification of the ranking order>",\n'
            '  "ranking": [index_of_best, index_of_second_best, ..., index_of_worst]\n'
            "}\n"
            "Where 'ranking' is a list of integers representing the 0-based indices of the candidates in order of quality.\n"
            f"Ensure the JSON is valid and contains exactly {len(keys)} unique indices from 0 to {len(keys)-1}."
        )

        user_prompt_lines = [
            f"Evaluation Criteria: {self.criteria}",
            f"Prompt: {record.prompt}",
        ]
        if record.context:
            user_prompt_lines.append(f"Context: {record.context}")
        if record.reference:
            user_prompt_lines.append(f"Reference Answer: {record.reference}")
        
        user_prompt_lines.append("\nCandidate Responses:")
        for idx, k in enumerate(keys):
            user_prompt_lines.append(f"[Candidate {idx}]: {completions[k]}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_lines)},
        ]

        raw_response = self.client.generate(messages)
        parsed = extract_json_response(raw_response)

        reasoning = parsed.get("reasoning", "")
        ranking = parsed.get("ranking", [])

        # Validate ranking list
        if not isinstance(ranking, list):
            raise ValueError(f"Ranking judge output 'ranking' must be a list. Got: {ranking}")
        
        expected_indices = set(range(len(keys)))
        actual_indices = set(ranking)
        
        if len(ranking) != len(keys) or expected_indices != actual_indices:
            raise ValueError(
                f"Ranking list {ranking} must contain exactly indices 0 to {len(keys)-1} without duplicates.\n"
                f"Raw response: {raw_response}"
            )

        # Map candidate indices back to their keys
        ranked_keys = [keys[idx] for idx in ranking]

        return EvalResult(
            record_id=record.id,
            task_type="ranking",
            reasoning=reasoning,
            judgement=ranked_keys,  # keys ordered from best to worst
            raw_response=raw_response,
            metadata={"candidate_keys_order": keys, "ranking_indices": ranking}
        )

class ScoringJudge(LLMBaseJudge):
    """Scores a single completion against a numerical criteria range (e.g. 1 to 5)."""

    def __init__(self, config: ModelConfig, criteria: str, min_score: float = 1.0, max_score: float = 5.0):
        super().__init__(config)
        self.criteria = criteria
        self.min_score = min_score
        self.max_score = max_score

    def evaluate(self, record: EvalRecord, completion: Optional[str] = None) -> EvalResult:
        comp_text = completion if completion is not None else record.completion
        if comp_text is None:
            raise ValueError(f"No completion text provided for record ID {record.id}")

        system_prompt = (
            f"You are an objective evaluation assistant. Your task is to score the provided response on a scale from {self.min_score} to {self.max_score} based on the specified criteria.\n"
            "First, reason step-by-step analyzing the response quality according to the criteria.\n"
            "Finally, output a JSON object in this exact format:\n"
            "{\n"
            '  "reasoning": "<detailed explanation of the reasoning and justification for the score>",\n'
            f'  "score": <float score between {self.min_score} and {self.max_score}>\n'
            "}\n"
            "Ensure the JSON is valid and the score is a number within the allowed range."
        )

        user_prompt_lines = [
            f"Evaluation Criteria: {self.criteria}",
            f"Prompt: {record.prompt}",
        ]
        if record.context:
            user_prompt_lines.append(f"Context: {record.context}")
        if record.reference:
            user_prompt_lines.append(f"Reference Answer: {record.reference}")
        user_prompt_lines.append(f"Response to Score: {comp_text}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_lines)},
        ]

        raw_response = self.client.generate(messages)
        parsed = extract_json_response(raw_response)

        reasoning = parsed.get("reasoning", "")
        score_val = parsed.get("score")
        
        try:
            score = float(score_val)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Scoring judge output 'score' could not be parsed as a float: {score_val}") from e

        if not (self.min_score <= score <= self.max_score):
            raise ValueError(f"Score {score} is outside the allowed range [{self.min_score}, {self.max_score}]")

        return EvalResult(
            record_id=record.id,
            task_type="scoring",
            reasoning=reasoning,
            judgement=score,
            raw_response=raw_response,
        )

class PairwiseJudge(LLMBaseJudge):
    """Compares two candidate completions, deciding which one is better, optionally mitigating bias."""

    def __init__(self, config: ModelConfig, criteria: str, mitigate_position_bias: bool = True):
        super().__init__(config)
        self.criteria = criteria
        self.mitigate_position_bias = mitigate_position_bias

    def _run_single_eval(
        self, record: EvalRecord, comp_a: str, comp_b: str, candidate_name_a: str = "A", candidate_name_b: str = "B"
    ) -> Tuple[str, str, str]:
        """Run comparison prompting where comp_a is presented as A and comp_b is presented as B."""
        system_prompt = (
            f"You are an objective evaluation assistant. Your task is to compare two candidate completions (Candidate A and Candidate B) and determine which one is better based on the specified criteria.\n"
            "First, reason step-by-step comparing the two candidates side-by-side. Critique their quality, accuracy, style, and tone based on the criteria.\n"
            "Finally, output a JSON object in this exact format:\n"
            "{\n"
            '  "reasoning": "<detailed comparison and step-by-step justification>",\n'
            f'  "winner": "<winner label: \'A\' if Candidate A is better, \'B\' if Candidate B is better, or \'tie\' if they are of equal quality>"\n'
            "}\n"
            "Ensure the JSON is valid and the winner is either 'A', 'B', or 'tie'."
        )

        user_prompt_lines = [
            f"Evaluation Criteria: {self.criteria}",
            f"Prompt: {record.prompt}",
        ]
        if record.context:
            user_prompt_lines.append(f"Context: {record.context}")
        if record.reference:
            user_prompt_lines.append(f"Reference Answer: {record.reference}")
        
        user_prompt_lines.append(f"\nCandidate A: {comp_a}")
        user_prompt_lines.append(f"Candidate B: {comp_b}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_lines)},
        ]

        raw_response = self.client.generate(messages)
        parsed = extract_json_response(raw_response)

        reasoning = parsed.get("reasoning", "")
        winner = str(parsed.get("winner", "")).strip()

        if winner not in ("A", "B", "tie"):
            if winner.lower() == "tie":
                winner = "tie"
            elif winner.lower() == "a":
                winner = "A"
            elif winner.lower() == "b":
                winner = "B"
            else:
                raise ValueError(f"Pairwise judge output 'winner' must be 'A', 'B', or 'tie'. Got: {winner}")

        return winner, reasoning, raw_response

    def evaluate_pair(self, record: EvalRecord, key_a: str, key_b: str) -> EvalResult:
        """Compare two completions from record.completions specified by key_a and key_b."""
        completions = record.completions
        if not completions or key_a not in completions or key_b not in completions:
            raise ValueError(f"Completions dictionary must contain keys '{key_a}' and '{key_b}' for pairwise evaluation.")

        comp_a = completions[key_a]
        comp_b = completions[key_b]

        # Call once (A vs B)
        winner_1, reasoning_1, raw_1 = self._run_single_eval(record, comp_a, comp_b)

        if not self.mitigate_position_bias:
            final_winner = key_a if winner_1 == "A" else (key_b if winner_1 == "B" else "tie")
            return EvalResult(
                record_id=record.id,
                task_type="pairwise",
                reasoning=reasoning_1,
                judgement=final_winner,
                raw_response=raw_1,
                metadata={"key_a": key_a, "key_b": key_b, "swap_run": False}
            )

        # Swapped call (B vs A)
        # B is presented as Candidate A, A is presented as Candidate B
        winner_2, reasoning_2, raw_2 = self._run_single_eval(record, comp_b, comp_a)

        # Resolve winner
        # winner_1 = A means comp_a won
        # winner_2 = A means comp_b won (since comp_b was Candidate A in run 2)
        # winner_1 = B means comp_b won
        # winner_2 = B means comp_a won
        choice_1 = "a" if winner_1 == "A" else ("b" if winner_1 == "B" else "tie")
        choice_2 = "b" if winner_2 == "A" else ("a" if winner_2 == "B" else "tie")

        combined_reasoning = (
            f"=== Pass 1 (A={key_a}, B={key_b}) ===\n"
            f"Decision: Candidate {winner_1} won.\n"
            f"Reasoning:\n{reasoning_1}\n\n"
            f"=== Pass 2 (A={key_b}, B={key_a}) ===\n"
            f"Decision: Candidate {winner_2} won.\n"
            f"Reasoning:\n{reasoning_2}"
        )

        if choice_1 == choice_2:
            final_winner = key_a if choice_1 == "a" else (key_b if choice_1 == "b" else "tie")
        else:
            # Position bias detected or inconsistent decisions
            final_winner = "tie"
            combined_reasoning += f"\n\n[System Alert] Position bias or inconsistency detected: Pass 1 voted {choice_1.upper()}, Pass 2 voted {choice_2.upper()}. Declaring a Tie."

        return EvalResult(
            record_id=record.id,
            task_type="pairwise",
            reasoning=combined_reasoning,
            judgement=final_winner,
            raw_response=f"Pass 1: {raw_1}\nPass 2: {raw_2}",
            metadata={"key_a": key_a, "key_b": key_b, "swap_run": True, "winner_1": winner_1, "winner_2": winner_2}
        )

    def evaluate(self, record: EvalRecord, completion: Optional[str] = None) -> EvalResult:
        # If evaluate is called directly, assume comparing first two completions in record.completions
        completions = record.completions
        if not completions or len(completions) < 2:
            raise ValueError("Pairwise evaluation requires record.completions containing at least 2 candidates.")
        
        keys = list(completions.keys())
        return self.evaluate_pair(record, keys[0], keys[1])
