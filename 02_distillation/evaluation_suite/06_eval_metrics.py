"""
Step 07: Evaluation Metrics Module for Teacher-Student Comparison
==================================================================
Implements the full evaluation matrix (model-agnostic, no LLM judge):

1. Structured Correctness - JSON/code validity, format checks
2. Task Success Score - Heuristic-based (no LLM)
3. Instruction Following Score - Length, format, refusal, language
4. Coverage Score - Completeness vs teacher output
5. Faithfulness Score - Grounding to context/teacher
6. Error Amplification - T✔S✔, T✔S✘, T✘S✔, T✘S✘ analysis
7. Context Grounding Ratio - Use of context terms

All metrics return 0.0-1.0 (or 0-2 for instruction following rubric)
"""

import re
import json
import math
import ast
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import Counter

# ============================================================================
# TEXT UTILITIES
# ============================================================================

_WORD_RE = re.compile(r"[a-z0-9']+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "once", "here", "there",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "don", "should", "now", "is", "am", "are", "was", "were", "be",
    "been", "being", "do", "does", "did", "having", "have", "has", "had", "this",
    "that", "these", "those", "of", "it", "its", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "our", "their"
}

def normalize(text: str) -> str:
    """Normalize text: lowercase, strip, remove quotes, collapse whitespace"""
    text = (text or "").lower().strip()
    # Remove surrounding quotes and common quote chars
    text = text.strip('"\'\'\"\u201c\u201d\u2018\u2019')
    # Collapse whitespace
    return " ".join(text.split())

def tokenize(text: str) -> List[str]:
    """Extract word tokens from text"""
    return _WORD_RE.findall(normalize(text))

def content_tokens(text: str) -> List[str]:
    """Extract content tokens (no stopwords, no digits)"""
    return [t for t in tokenize(text) if t not in STOPWORDS and not t.isdigit()]

def token_set(text: str) -> Set[str]:
    """Get unique content tokens as set"""
    return set(content_tokens(text))

def bow_vector(tokens: List[str]) -> Dict[str, float]:
    """Create normalized bag-of-words vector"""
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
    return {k: v / norm for k, v in tf.items()}

def cosine_sim(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts"""
    vec_a = bow_vector(content_tokens(text_a))
    vec_b = bow_vector(content_tokens(text_b))
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(val * vec_b.get(key, 0.0) for key, val in vec_a.items())
    return max(0.0, min(1.0, dot))

def word_count(text: str) -> int:
    """Count words in text"""
    return len(tokenize(text))

def ensure_string(val: Any) -> str:
    """Convert to string, handling lists/None"""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x)
    return str(val)


# ============================================================================
# METRIC 1: STRUCTURED CORRECTNESS
# ============================================================================

def check_json_validity(text: str) -> Tuple[bool, Optional[str]]:
    """
    Check if text contains valid JSON.
    Returns (is_valid, error_message)
    """
    # Try to find JSON in text
    text = text.strip()
    
    # Try direct parse
    try:
        json.loads(text)
        return True, None
    except:
        pass
    
    # Try to extract JSON block
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'(\{[\s\S]*\})',
        r'(\[[\s\S]*\])'
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json.loads(match.group(1))
                return True, None
            except json.JSONDecodeError as e:
                return False, str(e)
    
    return False, "No JSON found"


def check_code_syntax(text: str, language: str = "python") -> Tuple[bool, Optional[str]]:
    """
    Check if code has valid syntax.
    Currently supports Python. Returns (is_valid, error_message)
    """
    # Extract code block if present
    code_patterns = [
        rf'```{language}\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]
    
    code = text
    for pattern in code_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            code = match.group(1)
            break
    
    if language.lower() == "python":
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
    
    # For other languages, do basic bracket matching
    brackets = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for char in code:
        if char in brackets:
            stack.append(brackets[char])
        elif char in brackets.values():
            if not stack or stack.pop() != char:
                return False, "Unmatched brackets"
    
    return len(stack) == 0, "Unclosed brackets" if stack else None


def check_format_compliance(text: str, expected_format: str = None) -> Dict[str, Any]:
    """
    Check format compliance based on detected/expected format.
    Returns dict with format checks.
    """
    result = {
        "has_json": False,
        "json_valid": False,
        "has_code": False,
        "code_valid": False,
        "has_list": False,
        "has_headers": False,
        "has_table": False,
    }
    
    # Check for JSON
    if '{' in text or '[' in text:
        result["has_json"] = True
        result["json_valid"], _ = check_json_validity(text)
    
    # Check for code blocks
    if '```' in text or 'def ' in text or 'function ' in text:
        result["has_code"] = True
        result["code_valid"], _ = check_code_syntax(text)
    
    # Check for lists (numbered or bulleted)
    list_patterns = [
        r'^\s*[\-\*\•]\s+',  # Bullet points
        r'^\s*\d+[\.\)]\s+',  # Numbered
    ]
    for pattern in list_patterns:
        if re.search(pattern, text, re.MULTILINE):
            result["has_list"] = True
            break
    
    # Check for markdown headers
    if re.search(r'^#+\s+', text, re.MULTILINE):
        result["has_headers"] = True
    
    # Check for tables
    if '|' in text and re.search(r'\|.*\|', text):
        result["has_table"] = True
    
    return result


def structured_correctness_score(text: str, task_label: str) -> float:
    """
    Calculate structured correctness score based on task type.
    Returns 0.0-1.0
    """
    format_check = check_format_compliance(text)
    
    if task_label == "technical_code":
        # For code tasks, check syntax
        if format_check["has_code"]:
            return 1.0 if format_check["code_valid"] else 0.5
        # Check if any code-like content exists
        if re.search(r'def |class |function |=>|return ', text):
            valid, _ = check_code_syntax(text)
            return 1.0 if valid else 0.3
        return 0.2  # No code found for code task
    
    # For other tasks, check general structure
    score = 0.5  # Base score for having content
    
    if format_check["has_list"]:
        score += 0.15
    if format_check["has_headers"]:
        score += 0.15
    if format_check["has_json"] and format_check["json_valid"]:
        score += 0.2
    
    return min(1.0, score)


# ============================================================================
# METRIC 2: TASK SUCCESS SCORE (Heuristic, no LLM)
# ============================================================================

# Refusal phrases that indicate task failure
REFUSAL_PHRASES = [
    "i cannot", "i can't", "i'm unable", "i am unable",
    "i don't have", "i do not have",
    "as an ai", "as a language model",
    "i'm not able", "i am not able",
    "i apologize", "sorry, i cannot",
    "i'm sorry, but i", "i am sorry, but i",
    "it's not possible", "it is not possible",
    "i cannot provide", "i can't provide",
    "i cannot help", "i can't help",
    "i cannot assist", "i can't assist",
    "out of my scope", "beyond my capabilities",
    "i don't know", "i do not know",
]

def detect_refusal(text: str) -> bool:
    """Check if output contains refusal phrases"""
    text_lower = normalize(text)
    return any(phrase in text_lower for phrase in REFUSAL_PHRASES)


def task_success_score(
    output: str,
    instruction: str,
    teacher_output: str = None,
    task_label: str = "general_qa"
) -> float:
    """
    Heuristic-based task success score.
    Returns 0.0-1.0
    """
    if not output or not output.strip():
        return 0.0
    
    # Check for refusal
    if detect_refusal(output):
        return 0.1  # Refusal = mostly failed
    
    # Minimum length check
    if word_count(output) < 5:
        return 0.2  # Too short
    
    score = 0.5  # Base score
    
    # Check relevance to instruction
    instruction_relevance = cosine_sim(instruction, output)
    score += instruction_relevance * 0.3
    
    # Check similarity to teacher (if provided)
    if teacher_output:
        teacher_sim = cosine_sim(teacher_output, output)
        score += teacher_sim * 0.2
    
    # Task-specific bonuses
    if task_label == "technical_code":
        if '```' in output or 'def ' in output or 'function' in output:
            score += 0.1
    elif task_label == "math_logic":
        # Check for numbers/equations
        if re.search(r'\d+', output):
            score += 0.05
        if re.search(r'[=\+\-\*\/]', output):
            score += 0.05
    elif task_label == "classification_analysis":
        # Should have clear categories or analysis
        if format_check := check_format_compliance(output):
            if format_check.get("has_list"):
                score += 0.1
    elif task_label == "language_editing":
        # Should produce text, not refuse
        if word_count(output) >= 20:
            score += 0.1
    elif task_label == "creative_generative":
        # Creativity bonus for length and variety
        if word_count(output) >= 50:
            score += 0.1
    
    return min(1.0, score)


# ============================================================================
# METRIC 3: INSTRUCTION FOLLOWING SCORE
# ============================================================================

def detect_language(text: str) -> str:
    """Simple language detection based on character patterns"""
    if not text:
        return "unknown"
    
    # Count character types
    latin = len(re.findall(r'[a-zA-Z]', text))
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    arabic = len(re.findall(r'[\u0600-\u06ff]', text))
    cyrillic = len(re.findall(r'[\u0400-\u04ff]', text))
    
    max_count = max(latin, chinese, arabic, cyrillic)
    if max_count == 0:
        return "unknown"
    
    if chinese == max_count:
        return "chinese"
    if arabic == max_count:
        return "arabic"
    if cyrillic == max_count:
        return "russian"
    return "english"


def extract_length_constraint(instruction: str) -> Optional[int]:
    """Extract word/sentence limit from instruction if specified"""
    patterns = [
        r'(\d+)\s*words?\s*(?:or less|max|maximum|limit)?',
        r'(?:max|maximum|limit|under|less than)\s*(\d+)\s*words?',
        r'(\d+)\s*sentences?',
        r'in\s*(\d+)\s*(?:words?|sentences?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, instruction.lower())
        if match:
            return int(match.group(1))
    return None


def instruction_following_score(
    output: str,
    instruction: str,
    expected_language: str = "english"
) -> Dict[str, Any]:
    """
    Score instruction following on rubric (0-2).
    Returns dict with component scores and total.
    
    Rubric:
    - 2: Fully follows instructions
    - 1: Minor violation
    - 0: Major violation
    """
    result = {
        "length_score": 2,
        "format_score": 2,
        "refusal_score": 2,
        "language_score": 2,
        "violations": [],
    }
    
    # Check refusal (major violation)
    if detect_refusal(output):
        result["refusal_score"] = 0
        result["violations"].append("refusal_detected")
    
    # Check language
    detected_lang = detect_language(output)
    if expected_language and detected_lang != expected_language and detected_lang != "unknown":
        result["language_score"] = 0
        result["violations"].append(f"wrong_language:{detected_lang}")
    
    # Check length constraint
    length_limit = extract_length_constraint(instruction)
    if length_limit:
        actual_words = word_count(output)
        if actual_words > length_limit * 1.5:  # Major over
            result["length_score"] = 0
            result["violations"].append(f"length_exceeded:{actual_words}/{length_limit}")
        elif actual_words > length_limit * 1.1:  # Minor over
            result["length_score"] = 1
            result["violations"].append(f"length_slightly_exceeded:{actual_words}/{length_limit}")
    
    # Check format requirements
    format_keywords = {
        "json": lambda t: '{' in t or '[' in t,
        "list": lambda t: bool(re.search(r'^\s*[\-\*\•\d]', t, re.MULTILINE)),
        "table": lambda t: '|' in t,
        "code": lambda t: '```' in t or bool(re.search(r'def |function |class ', t)),
        "bullet": lambda t: bool(re.search(r'^\s*[\-\*\•]', t, re.MULTILINE)),
        "numbered": lambda t: bool(re.search(r'^\s*\d+[\.\)]', t, re.MULTILINE)),
    }
    
    instruction_lower = instruction.lower()
    for fmt, checker in format_keywords.items():
        if fmt in instruction_lower:
            if not checker(output):
                result["format_score"] = 1
                result["violations"].append(f"missing_format:{fmt}")
                break
    
    # Calculate total (0-8 scaled to 0-2)
    total = (
        result["length_score"] +
        result["format_score"] +
        result["refusal_score"] +
        result["language_score"]
    )
    result["total_score"] = total / 4  # Normalize to 0-2
    result["normalized_score"] = total / 8  # Normalize to 0-1
    
    return result


# ============================================================================
# METRIC 4: COVERAGE / COMPLETENESS SCORE
# ============================================================================

def extract_key_elements(text: str) -> Set[str]:
    """
    Extract key elements from text (noun phrases, entities, key terms).
    Uses simple heuristics without NLP libraries.
    """
    tokens = content_tokens(text)
    
    # Get bi-grams for phrases
    bigrams = set()
    for i in range(len(tokens) - 1):
        bigrams.add(f"{tokens[i]}_{tokens[i+1]}")
    
    return set(tokens) | bigrams


def coverage_score(student_output: str, teacher_output: str) -> Dict[str, Any]:
    """
    Calculate coverage: what fraction of teacher's key elements
    are present in student output.
    
    Returns dict with score and details.
    """
    teacher_elements = extract_key_elements(teacher_output)
    student_elements = extract_key_elements(student_output)
    
    if not teacher_elements:
        return {
            "coverage_score": 1.0,
            "elements_covered": 0,
            "elements_total": 0,
            "missing_elements": []
        }
    
    covered = teacher_elements & student_elements
    missing = teacher_elements - student_elements
    
    # Limit missing to top items by removing bigrams if too many
    missing_list = sorted(missing)[:20]
    
    return {
        "coverage_score": len(covered) / len(teacher_elements),
        "elements_covered": len(covered),
        "elements_total": len(teacher_elements),
        "missing_elements": missing_list
    }


# ============================================================================
# METRIC 4b: CONCISENESS SCORE
# ============================================================================

def conciseness_score(student_output: str, teacher_output: str) -> Dict[str, Any]:
    """
    Penalize verbose outputs. Compare length ratio.
    Ideal ratio is 0.8-1.2 (similar length to teacher).
    
    Returns dict with score and details.
    """
    student_words = word_count(student_output)
    teacher_words = word_count(teacher_output)
    
    if teacher_words == 0:
        return {
            "conciseness_score": 1.0 if student_words < 50 else 0.5,
            "length_ratio": 0.0,
            "student_words": student_words,
            "teacher_words": teacher_words
        }
    
    ratio = student_words / teacher_words
    
    # Score based on ratio:
    # 0.8 - 1.2 = perfect (1.0)
    # 0.5 - 2.0 = okay (0.7)
    # < 0.5 or > 3.0 = poor (0.3)
    # > 5.0 = very poor (0.1)
    if 0.8 <= ratio <= 1.2:
        score = 1.0
    elif 0.5 <= ratio <= 2.0:
        score = 0.8
    elif 0.3 <= ratio <= 3.0:
        score = 0.6
    elif 0.2 <= ratio <= 5.0:
        score = 0.4
    else:
        score = 0.2
    
    return {
        "conciseness_score": score,
        "length_ratio": round(ratio, 2),
        "student_words": student_words,
        "teacher_words": teacher_words
    }


# ============================================================================
# METRIC 5: FAITHFULNESS / HALLUCINATION
# ============================================================================

def faithfulness_score(
    student_output: str,
    teacher_output: str = None,
    context: str = None,
    beta: float = 0.6
) -> Dict[str, Any]:
    """
    Calculate faithfulness: agreement with teacher and context.
    beta * cos(student, teacher) + (1-beta) * cos(student, context)
    
    Returns dict with score and components.
    """
    student_vec = bow_vector(content_tokens(student_output))
    
    teacher_sim = 0.0
    context_sim = 0.0
    
    if teacher_output:
        teacher_vec = bow_vector(content_tokens(teacher_output))
        if student_vec and teacher_vec:
            teacher_sim = sum(
                val * teacher_vec.get(key, 0.0)
                for key, val in student_vec.items()
            )
    
    if context:
        context_vec = bow_vector(content_tokens(context))
        if student_vec and context_vec:
            context_sim = sum(
                val * context_vec.get(key, 0.0)
                for key, val in student_vec.items()
            )
    
    # Calculate combined score
    if teacher_output and context:
        combined = beta * teacher_sim + (1 - beta) * context_sim
    elif teacher_output:
        combined = teacher_sim
    elif context:
        combined = context_sim
    else:
        combined = 0.0
    
    return {
        "faithfulness_score": max(0.0, min(1.0, combined)),
        "teacher_similarity": teacher_sim,
        "context_similarity": context_sim,
    }


def hallucination_score(student_output: str, context: str) -> Dict[str, Any]:
    """
    Calculate hallucination rate based on context grounding.
    Higher score = more hallucination.
    """
    if not context:
        return {
            "hallucination_score": 0.0,
            "grounded_ratio": 1.0,
            "ungrounded_terms": []
        }
    
    context_terms = token_set(context)
    student_terms = token_set(student_output)
    
    if not student_terms:
        return {
            "hallucination_score": 0.0,
            "grounded_ratio": 1.0,
            "ungrounded_terms": []
        }
    
    grounded = student_terms & context_terms
    ungrounded = student_terms - context_terms
    
    grounded_ratio = len(grounded) / len(student_terms) if student_terms else 1.0
    hallucination_rate = 1.0 - grounded_ratio
    
    return {
        "hallucination_score": hallucination_rate,
        "grounded_ratio": grounded_ratio,
        "ungrounded_terms": sorted(ungrounded)[:15]
    }


# ============================================================================
# METRIC 6: ERROR AMPLIFICATION ANALYSIS
# ============================================================================

def error_amplification_category(
    teacher_success: bool,
    student_success: bool
) -> str:
    """
    Categorize error amplification.
    
    Returns one of:
    - "both_correct": T✔ S✔ - OK
    - "distillation_loss": T✔ S✘ - Student failed where teacher succeeded
    - "student_hallucination": T✘ S✔ - Student succeeded where teacher failed
    - "dataset_issue": T✘ S✘ - Both failed
    """
    if teacher_success and student_success:
        return "both_correct"
    elif teacher_success and not student_success:
        return "distillation_loss"
    elif not teacher_success and student_success:
        return "student_hallucination"
    else:
        return "dataset_issue"


def evaluate_success_threshold(
    output: str,
    instruction: str,
    context: str = None,
    threshold: float = 0.5
) -> bool:
    """
    Determine if output is successful based on multiple heuristics.
    """
    # Quick fails
    if not output or not output.strip():
        return False
    if detect_refusal(output):
        return False
    if word_count(output) < 5:
        return False
    
    # Check instruction relevance
    relevance = cosine_sim(instruction, output)
    if relevance < 0.1:
        return False
    
    # Check context grounding if available
    if context:
        context_relevance = cosine_sim(context, output)
        combined = (relevance + context_relevance) / 2
        return combined >= threshold
    
    return relevance >= threshold


# ============================================================================
# METRIC 7: CONTEXT GROUNDING RATIO
# ============================================================================

def context_grounding_score(student_output: str, context: str) -> Dict[str, Any]:
    """
    Measure how well student uses provided context.
    
    Context_Grounding = |Context_terms ∩ Output_terms| / |Context_terms|
    """
    if not context:
        return {
            "context_grounding_score": 1.0,  # No context = fully grounded
            "context_terms_used": 0,
            "context_terms_total": 0,
            "unused_context_terms": []
        }
    
    context_terms = token_set(context)
    student_terms = token_set(student_output)
    
    if not context_terms:
        return {
            "context_grounding_score": 1.0,
            "context_terms_used": 0,
            "context_terms_total": 0,
            "unused_context_terms": []
        }
    
    used = context_terms & student_terms
    unused = context_terms - student_terms
    
    return {
        "context_grounding_score": len(used) / len(context_terms),
        "context_terms_used": len(used),
        "context_terms_total": len(context_terms),
        "unused_context_terms": sorted(unused)[:15]
    }


# ============================================================================
# MASTER EVALUATION FUNCTION
# ============================================================================

def evaluate_single_output(
    instruction: str,
    student_output: str,
    teacher_output: str = None,
    context: str = None,
    task_label: str = "general_qa"
) -> Dict[str, Any]:
    """
    Run all evaluation metrics on a single output.
    
    Returns comprehensive evaluation dict.
    """
    # Ensure strings
    instruction = ensure_string(instruction)
    student_output = ensure_string(student_output)
    teacher_output = ensure_string(teacher_output) if teacher_output else ""
    context = ensure_string(context) if context else ""
    
    # 1. Structured Correctness
    struct_score = structured_correctness_score(student_output, task_label)
    
    # 2. Task Success
    task_score = task_success_score(
        student_output, instruction, teacher_output, task_label
    )
    
    # 3. Instruction Following
    instr_follow = instruction_following_score(student_output, instruction)
    
    # 4. Coverage
    coverage = coverage_score(student_output, teacher_output) if teacher_output else {
        "coverage_score": 0.0, "elements_covered": 0, "elements_total": 0
    }
    
    # 5. Faithfulness
    faith = faithfulness_score(student_output, teacher_output, context)
    
    # 6. Hallucination
    hallu = hallucination_score(student_output, context) if context else {
        "hallucination_score": 0.0, "grounded_ratio": 1.0
    }
    
    # 7. Context Grounding
    ctx_ground = context_grounding_score(student_output, context) if context else {
        "context_grounding_score": 1.0
    }
    
    # 8. Conciseness (penalize verbose outputs)
    concise = conciseness_score(student_output, teacher_output) if teacher_output else {
        "conciseness_score": 1.0, "length_ratio": 1.0
    }
    
    # Calculate overall score
    # Positive metrics (higher is better)
    positive_scores = [
        struct_score,
        task_score,
        instr_follow["normalized_score"],
        coverage["coverage_score"],
        faith["faithfulness_score"],
        ctx_ground["context_grounding_score"],
        concise["conciseness_score"],  # Added conciseness
    ]
    
    # Negative metrics (lower is better, so we invert)
    negative_scores = [
        hallu["hallucination_score"],
    ]
    
    # Overall = mean(positives) * (1 - mean(negatives))
    mean_positive = sum(positive_scores) / len(positive_scores)
    mean_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0
    overall = mean_positive * (1 - mean_negative)
    
    return {
        # Individual scores
        "structured_correctness": round(struct_score, 4),
        "task_success": round(task_score, 4),
        "instruction_following": round(instr_follow["normalized_score"], 4),
        "instruction_following_rubric": round(instr_follow["total_score"], 2),
        "coverage": round(coverage["coverage_score"], 4),
        "faithfulness": round(faith["faithfulness_score"], 4),
        "hallucination": round(hallu["hallucination_score"], 4),
        "context_grounding": round(ctx_ground["context_grounding_score"], 4),
        "conciseness": round(concise["conciseness_score"], 4),
        
        # Overall
        "overall_score": round(overall, 4),
        
        # Detailed components
        "details": {
            "instruction_violations": instr_follow.get("violations", []),
            "teacher_similarity": round(faith.get("teacher_similarity", 0), 4),
            "context_similarity": round(faith.get("context_similarity", 0), 4),
            "elements_covered": coverage.get("elements_covered", 0),
            "elements_total": coverage.get("elements_total", 0),
            "length_ratio": concise.get("length_ratio", 1.0),
        }
    }


def compare_teacher_student(
    instruction: str,
    teacher_output: str,
    student_output: str,
    context: str = None,
    task_label: str = "general_qa"
) -> Dict[str, Any]:
    """
    Compare teacher and student outputs, including error amplification.
    """
    # Evaluate both
    teacher_eval = evaluate_single_output(
        instruction, teacher_output, None, context, task_label
    )
    student_eval = evaluate_single_output(
        instruction, student_output, teacher_output, context, task_label
    )
    
    # Error amplification
    teacher_success = evaluate_success_threshold(teacher_output, instruction, context)
    student_success = evaluate_success_threshold(student_output, instruction, context)
    error_cat = error_amplification_category(teacher_success, student_success)
    
    # Calculate degradation (how much worse is student vs teacher)
    degradation = {
        "overall": round(teacher_eval["overall_score"] - student_eval["overall_score"], 4),
        "task_success": round(teacher_eval["task_success"] - student_eval["task_success"], 4),
        "instruction_following": round(
            teacher_eval["instruction_following"] - student_eval["instruction_following"], 4
        ),
    }
    
    return {
        "teacher": teacher_eval,
        "student": student_eval,
        "error_category": error_cat,
        "degradation": degradation,
        "student_wins": student_eval["overall_score"] > teacher_eval["overall_score"],
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test cases
    print("=" * 60)
    print("EVALUATION METRICS MODULE TEST")
    print("=" * 60)
    
    # Test 1: Code task
    print("\n--- Test 1: Code Task ---")
    result = evaluate_single_output(
        instruction="Write a Python function to calculate factorial",
        student_output="""
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```
""",
        teacher_output="""
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
""",
        task_label="technical_code"
    )
    print(f"Scores: {result}")
    
    # Test 2: Refusal detection
    print("\n--- Test 2: Refusal ---")
    result = evaluate_single_output(
        instruction="Explain quantum computing",
        student_output="I'm sorry, but I cannot provide information about that topic.",
        teacher_output="Quantum computing uses quantum bits or qubits...",
        task_label="general_qa"
    )
    print(f"Task Success: {result['task_success']}")
    print(f"Instruction Following: {result['instruction_following']}")
    
    # Test 3: Compare teacher vs student
    print("\n--- Test 3: Teacher vs Student Comparison ---")
    comparison = compare_teacher_student(
        instruction="Summarize the key benefits of exercise",
        teacher_output="Exercise improves cardiovascular health, builds muscle strength, enhances mental well-being, and helps maintain healthy weight.",
        student_output="Exercise is good for your heart and muscles.",
        task_label="language_editing"
    )
    print(f"Teacher Overall: {comparison['teacher']['overall_score']}")
    print(f"Student Overall: {comparison['student']['overall_score']}")
    print(f"Error Category: {comparison['error_category']}")
    print(f"Degradation: {comparison['degradation']}")
    
    print("\n✓ All tests completed!")
