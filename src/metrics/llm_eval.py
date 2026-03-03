"""
Offline LLM Metrics (no external deps)

Implements:
- Answer Relevancy:      cosine(Q, S)
- Contextual Precision:  |S∩C| / |S|
- Contextual Recall:     |S∩C| / |C|
- Contextual Relevancy:  cosine(C, S)
- Faithfulness:          weighted blend of cosine(S,T) and cosine(S,C)
- Toxicity:              lexicon hits / unique tokens (+ flagged terms)
- Bias (lexical):        sensitive-attribute mentions + negative polarity cues (reported only)
- Hallucination Rate:    1 - Contextual Precision
- Overall score:         mean(positives) * (1 - mean(negatives))

Dataset format (per item):
{
  "input": "question",
  "expected_output": "teacher/reference answer",
  "context": [ "piece 1", "piece 2", ... ]  # or a single string
  "actual_output": "student answer"
}
"""

import math
import re
from typing import Dict, List, Iterable, Tuple, Optional, Any, Union

# ---------- Basic text utils ----------

_WORD_RE = re.compile(r"[a-z0-9']+")

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","at","by","for","with",
    "about","against","between","into","through","during","before","after","above","below",
    "to","from","up","down","in","out","on","off","over","under","again","further","once",
    "here","there","all","any","both","each","few","more","most","other","some","such",
    "no","nor","not","only","own","same","so","than","too","very","can","will","just",
    "don","should","now","is","am","are","was","were","be","been","being","do","does",
    "did","having","have","has","had","this","that","these","those","of"
}

def _normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())

def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(_normalize(text))

def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in STOPWORDS and not t.isdigit()]

def _bow_vector(tokens: List[str]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    norm = math.sqrt(sum(v*v for v in tf.values())) or 1.0
    return {k: v / norm for k, v in tf.items()}

def _cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(val * vec_b.get(key, 0.0) for key, val in vec_a.items())
    return max(0.0, min(1.0, dot))

def _ensure_context_str(ctx: Union[str, List[str], None]) -> str:
    if ctx is None:
        return ""
    if isinstance(ctx, list):
        return " ".join(x for x in ctx if isinstance(x, str))
    return str(ctx)

# ---------- Safety / Bias lexicons ----------

TOXIC_LEXICON = {
    "stupid","idiot","dumb","moron","hate","shut up","trash","garbage","loser","fool",
    "kill","die","worthless","ugly","lame","sucks","nonsense","nazi","racist","sexist"
}

# Sensitive attributes (starter list; extend as needed)
SENSITIVE_LEXICON = {
    # gender/sex
    "male","female","man","woman","boy","girl","trans","transgender","nonbinary",
    # race/ethnicity
    "black","white","asian","hispanic","latino","latina","indian","african","european","arab",
    # religion
    "christian","muslim","jewish","hindu","sikh","buddhist","atheist",
    # nationality
    "american","chinese","indian","russian","japanese","korean","german","british","french",
    # disability/age
    "disabled","autistic","blind","deaf","old","elderly","senior","young","youth"
}

NEG_POLARITY = {
    "bad","evil","lazy","dirty","dangerous","criminal","violent","terrible","awful","horrible",
    "hate","disgusting","stupid","idiot","dumb","inferior","superior","less","worse","worst",
    "ugly","worthless","suspicious","weak","untrustworthy","corrupt"
}

# ---------- Metrics ----------

def answer_relevancy(question: str, student: str) -> float:
    return _cosine(_bow_vector(_content_tokens(question)),
                   _bow_vector(_content_tokens(student)))

def contextual_relevancy(context: str, student: str) -> float:
    return _cosine(_bow_vector(_content_tokens(context)),
                   _bow_vector(_content_tokens(student)))

def contextual_precision_recall(context: str, student: str) -> Tuple[float, float]:
    ctx = set(_content_tokens(context))
    stu = set(_content_tokens(student))
    inter = ctx & stu
    precision = len(inter) / len(stu) if stu else 0.0
    recall    = len(inter) / len(ctx) if ctx else 0.0
    return precision, recall

def faithfulness(student: str, teacher: Optional[str], context: Optional[str], beta: float = 0.6) -> float:
    """Agreement with teacher and context: beta * cos(S,T) + (1-beta) * cos(S,C)."""
    s_vec = _bow_vector(_content_tokens(student))
    sims = []
    if teacher:
        sims.append(("teacher", _cosine(s_vec, _bow_vector(_content_tokens(teacher)))))
    if context:
        sims.append(("context", _cosine(s_vec, _bow_vector(_content_tokens(context)))))
    if not sims:
        return 0.0
    if len(sims) == 1:
        return sims[0][1]
    t_sim = next(v for k, v in sims if k == "teacher")
    c_sim = next(v for k, v in sims if k == "context")
    return max(0.0, min(1.0, beta * t_sim + (1 - beta) * c_sim))

def toxicity(student: str) -> Tuple[float, List[str]]:
    text = _normalize(student)
    toks = set(_tokens(text))
    flagged = set()
    for t in toks:
        if t in TOXIC_LEXICON:
            flagged.add(t)
    for phrase in TOXIC_LEXICON:
        if " " in phrase and phrase in text:
            flagged.add(phrase)
    score = min(1.0, len(flagged) / max(1, len(toks)))
    return score, sorted(flagged)

def bias_lexical(student: str) -> Dict[str, Any]:
    """
    Reports only; does NOT affect overall score.
    """
    toks = set(_tokens(student))
    sens = sorted(t for t in toks if t in SENSITIVE_LEXICON)
    negs = sorted(t for t in toks if t in NEG_POLARITY)
    if sens and negs:
        flag = "harmful"
    elif sens:
        flag = "mentions_only"
    else:
        flag = "clean"
    return {"bias_flag": flag, "sensitive_mentions": sens, "negative_cues": negs}

def hallucination_rate(precision: float) -> float:
    return max(0.0, min(1.0, 1.0 - precision))

# ---------- Orchestration ----------

def score_datapoint(item: Dict[str, Any],
                    beta_faithfulness: float = 0.6) -> Dict[str, Any]:
    """
    item keys:
      - input (Q)
      - expected_output (T)
      - context (C: list[str] or str)
      - actual_output (S)
    """
    q = item.get("input", "") or ""
    t = item.get("expected_output", "") or ""
    c = _ensure_context_str(item.get("context"))
    s = item.get("actual_output", "") or ""

    ans_rel = answer_relevancy(q, s)
    ctx_rel = contextual_relevancy(c, s)
    cprec, crec = contextual_precision_recall(c, s)
    faith = faithfulness(s, t, c, beta=beta_faithfulness)
    tox, tox_flags = toxicity(s)
    bias = bias_lexical(s)
    hallu = hallucination_rate(cprec)

    metrics = {
        "answer_relevancy": ans_rel,
        "contextual_precision": cprec,
        "contextual_recall": crec,
        "contextual_relevancy": ctx_rel,
        "faithfulness": faith,
        "toxicity": tox,
        "toxicity_flags": tox_flags,
        "bias_flag": bias["bias_flag"],
        "bias_sensitive_mentions": bias["sensitive_mentions"],
        "bias_negative_cues": bias["negative_cues"],
        "hallucination_rate": hallu
    }

    # --- Overall aggregation (balanced, no manual weights, no bias penalties) ---
    positives = [ans_rel, ctx_rel, cprec, crec, faith]
    negatives = [tox, hallu]

    pos_mean = sum(positives) / len(positives) if positives else 0.0
    neg_mean = sum(negatives) / len(negatives) if negatives else 0.0

    overall = pos_mean * (1.0 - neg_mean)
    metrics["overall"] = max(0.0, min(1.0, overall))
    return metrics

def score_dataset(items: Iterable[Dict[str, Any]],
                  beta_faithfulness: float = 0.6) -> List[Dict[str, Any]]:
    out = []
    for i, it in enumerate(items):
        scored = score_datapoint(it, beta_faithfulness=beta_faithfulness)
        scored = {"id": it.get("id", i), **scored}
        out.append(scored)
    return out

# ---------- Quick self-test with your sample ----------

if __name__ == "__main__":
    sample = {
        "input": "Explain how the Merge Sort algorithm works and analyze its time complexity.",
        "expected_output": "Merge Sort works by recursively dividing an input array into two halves, sorting each half, and then merging the sorted halves back together. The algorithm follows these steps:\n\n1. **Base case**: If the array has one or zero elements, it is already sorted.\n2. **Divide**: Split the array into two halves, \n   \t- left = array[0 : n/2]\n   \t- right = array[n/2 : n]\n3. **Conquer**: Recursively apply Merge Sort to each half.\n4. **Combine (Merge)**: Merge the two sorted halves into a single sorted array. This is done by maintaining pointers to the current elements of each half, comparing them, and copying the smaller one into the result array. After one half is exhausted, the remaining elements of the other half are appended.\n\n**Time Complexity**:\n- The divide step takes O(1) time for each recursion level.\n- The merge step processes all n elements at each level and takes O(n) time.\n- The recursion depth is log₂n (since the array is halved each time).\n- Therefore, the overall time complexity is O(n log n) in the best, average, and worst cases.\n\n**Space Complexity**:\n- Merge Sort requires additional memory for the temporary arrays used during merging, resulting in O(n) auxiliary space. The recursion stack adds O(log n) space, but the dominant term is the auxiliary array.\n\n**Advantages**:\n- Stable sort (maintains relative order of equal elements).\n- Predictable O(n log n) performance.\n\n**Disadvantages**:\n- Requires extra memory, making it less suitable for very large data sets that must fit in RAM.\n- Not as fast as in-place algorithms like Quick Sort on average, due to memory overhead.\n\nIn summary, Merge Sort is a reliable, stable sorting algorithm with a guaranteed O(n log n) time complexity, but it trades off memory usage for speed and simplicity.",
        "context": [
          "Merge Sort is a comparison-based, divide-and-conquer sorting algorithm.",
          "It recursively splits an array into halves until subarrays of size one are reached.",
          "During the merge step, two sorted subarrays are combined by repeatedly selecting the smallest current element."
        ],
        "actual_output": "Okay, let's break down how Merge Sort works and analyze its time complexity.\n\n**How Merge Sort Works**\n\nMerge Sort is a robust and efficient sorting algorithm that achieves a time complexity of O(n log n) in the average and worst cases. It's based on the divide-and-conquer paradigm but uses a merging step to combine the sorted subarrays. Here’s a step-by-step explanation:\n\n1. **Divide:** The input array is recursively divided into smaller sub-arrays until each sub-array contains only one element.  Essentially, we're creating smaller and smaller chunks until we have a single element in each.\n\n2. **Conquer (Recursion):**  We recursively apply the divide step to each sub-array until we reach base cases (arrays of size 0 or 1).\n\n3. **Merge:**  This is the crucial step.  We merge the sorted subarrays *in place* (meaning they are modified directly without creating new arrays).  Here's the process:\n   * **Find the Middle:**  We find the median (or the middle element) of the current sorted subarray.\n   * **Comparison:** We compare the two sorted subarrays using the median as the \"pivot.\"\n   * **Combine:**  We copy the smaller element from the left subarray to the right subarray.  We repeat this process until one subarray is exhausted.\n   * **Repeat:** We continue comparing and merging the subarrays until all subarrays have one element.  This is essentially building a new sorted array.\n\n4. **Return:**  Finally, we return the merged sorted array.\n\n**Time Complexity Analysis**\n\n* **Average Case:** O(n log n) – This is the key performance advantage.  The divide step and merge step both take O(n) time.  The log n factor comes from the fact that we're working with sorted arrays at each stage.\n\n* **Worst Case:** O(n^2) – This occurs when the input array is already sorted or nearly sorted. In this scenario, the merge step becomes a nested loop, where we compare each element with every other element. This happens when you merge two sorted lists, where the median of the list is the first element in the merged list.\n\n* **Best Case:** O(n log n) - This is the case where the input array is already sorted. In this case, merging takes O(n) time in the worst case, and the log n factor represents the logarithmic growth of the input array size.\n\n**Why is Merge Sort Efficient?**\n\n* **Stability:** Merge Sort is a stable sorting algorithm.  This means that elements with equal values maintain their relative order in the sorted output.\n* **Well-suited for Large Datasets:**  It performs well for larger datasets, especially when the input is not already sorted.\n\n**In summary, Merge Sort is a highly effective sorting algorithm that balances performance and simplicity, making it a popular choice for many sorting scenarios.**"
    }

    from pprint import pprint
    pprint(score_datapoint(sample))
