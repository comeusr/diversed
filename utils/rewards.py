"""
Reward functions for different datasets used in DIVERSED training.
"""

import re
import evaluate
import sacrebleu
from typing import List, Dict, Any, Union, Optional


def extract_first_answer_block(text: str) -> str:
    """
    Extract the first answer block from a text.
    
    Args:
        text: Input text.
        
    Returns:
        First answer block.
    """
    split_marker = "Question:"
    if split_marker in text:
        return text.split(split_marker, 1)[0].strip()
    return text.strip()


def find_answer(text: str) -> Union[int, str]:
    """
    Find the answer in a text for GSM8K.
    
    Args:
        text: Input text.
        
    Returns:
        Extracted answer.
    """
    match = re.search(r"###\s*(-?\d+)", text.replace(",", ""))
    if match:
        return round(float(match.group(1)))
    else:
        all_m = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", text.replace(",", ""))
        if all_m:
            return round(float(all_m[-1]))
    return "No answer found"


def gsm8k_reward_func(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """
    Reward function for GSM8K dataset.
    
    Args:
        completions: List of model completions.
        ground_truth: List of ground truth answers.
        **kwargs: Additional arguments.
        
    Returns:
        List of rewards (1.0 for correct answers, 0.0 for incorrect).
    """
    contents = [find_answer(extract_first_answer_block(completion)) for completion in completions]
    ground_truth = [find_answer(truth) for truth in ground_truth]
    
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]


def cnndm_find_answer(text: str) -> str:
    """
    Extract the summary from CNN/DailyMail format.
    
    Args:
        text: Input text.
        
    Returns:
        Extracted summary.
    """
    return re.split(r"\n\nArticle:", text)[0].strip()


def cnndm_reward_func(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """
    Reward function for CNN/DailyMail dataset.
    
    Args:
        completions: List of model completions.
        ground_truth: List of ground truth summaries.
        **kwargs: Additional arguments.
        
    Returns:
        List of ROUGE-2 scores.
    """
    rouge = evaluate.load("rouge")
    
    results = []
    for completion, gt in zip(completions, ground_truth):
        completion = cnndm_find_answer(completion)
        results.append(rouge.compute(predictions=[completion], references=[gt])['rouge2'])
    
    return results


def xsum_find_answer(text: str) -> str:
    """
    Extract the summary from XSum format.
    
    Args:
        text: Input text.
        
    Returns:
        Extracted summary.
    """
    return re.split(r"\n\nDocument:", text)[0].strip()


def truncate_to_n_words(text: str, n: int = 32) -> str:
    """
    Truncate text to the first n words.
    
    Args:
        text: The text to truncate.
        n: Number of words to keep.
        
    Returns:
        Truncated text containing only the first n words.
    """
    words = text.split()
    if len(words) <= n:
        return text
    return ' '.join(words[:n])


def xsum_reward_func(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """
    Reward function for XSum dataset.
    
    Args:
        completions: List of model completions.
        ground_truth: List of ground truth summaries.
        **kwargs: Additional arguments.
        
    Returns:
        List of ROUGE-2 scores.
    """
    rouge = evaluate.load("rouge")
    
    results = []
    for completion, gt in zip(completions, ground_truth):
        completion = xsum_find_answer(completion)
        # Truncate to first 32 words for XSum
        completion = truncate_to_n_words(completion, 32)
        results.append(rouge.compute(predictions=[completion], references=[gt])['rouge2'])
    
    return results


def wmt_find_answer(text: str) -> str:
    """
    Extract the translation from WMT format.
    
    Args:
        text: Input text.
        
    Returns:
        Extracted translation.
    """
    return re.split(r"\n\nGerman:", text)[0].strip()


def wmt_reward_func(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """
    Reward function for WMT dataset.
    
    Args:
        completions: List of model completions.
        ground_truth: List of ground truth translations.
        **kwargs: Additional arguments.
        
    Returns:
        List of BLEU scores.
    """
    contents = [wmt_find_answer(complete) for complete in completions]
    rewards = []
    
    for content, gt in zip(contents, ground_truth):
        rewards.append(sacrebleu.sentence_bleu(content, [gt]).score)
    
    return rewards


def humaneval_clean_pred(completion: str) -> str:
    """
    Clean HumanEval prediction.
    
    Args:
        completion: Model completion.
        
    Returns:
        Cleaned completion.
    """
    # Extract the function definition
    lines = completion.strip().split('\n')
    cleaned_lines = []
    
    # Skip any non-code lines at the beginning
    started = False
    for line in lines:
        if not started and (line.startswith('def ') or line.strip() == ''):
            started = True
        if started:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def check_correctness(completion: str, problem: Dict[str, Any]) -> bool:
    """
    Check if a HumanEval solution is correct.
    
    Args:
        completion: Model completion.
        problem: Problem definition.
        
    Returns:
        True if the solution is correct, False otherwise.
    """
    try:
        # Create a namespace for execution
        namespace = {}
        
        # Execute the completion
        exec(completion, namespace)
        
        # Get the function name from the prompt
        func_name = problem['entry_point']
        
        # Check if the function exists in the namespace
        if func_name not in namespace:
            return False
        
        # Get the function
        func = namespace[func_name]
        
        # Execute test cases
        for test_case in problem['test_list']:
            # Extract test case
            test_code = test_case.strip()
            
            # Execute test case
            try:
                exec(f"assert {test_code}", namespace)
            except AssertionError:
                return False
            except Exception:
                return False
        
        return True
    except Exception:
        return False


def humaneval_reward_func(completions: List[str], ground_truth: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function for HumanEval dataset.
    
    Args:
        completions: List of model completions.
        ground_truth: List of ground truth solutions.
        **kwargs: Additional arguments.
        
    Returns:
        List of rewards (1.0 for correct solutions, 0.0 for incorrect).
    """
    # If problem information is provided in kwargs, use it
    if 'problem' in kwargs:
        problems = kwargs['problem']
    else:
        problems = ground_truth
    
    results = []
    for completion, problem in zip(completions, problems):
        cleaned_completion = humaneval_clean_pred(completion)
        results.append(1.0 if check_correctness(cleaned_completion, problem) else 0.0)
    
    return results


def mbpp_clean_pred(completion: str) -> str:
    """
    Clean MBPP prediction.
    
    Args:
        completion: Model completion.
        
    Returns:
        Cleaned completion.
    """
    # Similar to HumanEval cleaning
    return humaneval_clean_pred(completion)


def check_correctness_mbpp(completion: str, problem: Dict[str, Any]) -> bool:
    """
    Check if an MBPP solution is correct.
    
    Args:
        completion: Model completion.
        problem: Problem definition.
        
    Returns:
        True if the solution is correct, False otherwise.
    """
    try:
        # Create a namespace for execution
        namespace = {}
        
        # Execute the completion
        exec(completion, namespace)
        
        # Execute test cases
        for test_case in problem['test_list']:
            # Extract test case
            test_code = test_case.strip()
            
            # Execute test case
            try:
                exec(f"assert {test_code}", namespace)
            except AssertionError:
                return False
            except Exception:
                return False
        
        return True
    except Exception:
        return False


def mbpp_reward_func(completions: List[str], ground_truth: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function for MBPP dataset.
    
    Args:
        completions: List of model completions.
        ground_truth: List of ground truth solutions.
        **kwargs: Additional arguments.
        
    Returns:
        List of rewards (1.0 for correct solutions, 0.0 for incorrect).
    """
    # If problem information is provided in kwargs, use it
    if 'problem' in kwargs:
        problems = kwargs['problem']
    else:
        problems = ground_truth
    
    results = []
    for completion, problem in zip(completions, problems):
        cleaned_completion = mbpp_clean_pred(completion)
        results.append(1.0 if check_correctness_mbpp(cleaned_completion, problem) else 0.0)
    
    return results
