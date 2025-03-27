

from rouge_score import rouge_scorer


def rougeL(references, predictions, **kwargs):
    scores = []
    for pred, ref in zip(predictions, references):
        pred = pred.strip()
        ref = ref.strip()
        pred = " ".join(pred.split())
        ref = " ".join(ref.split())
        scores.append(float(rouge_scorer.RougeScorer(['rougeL']).score(ref, pred)['rougeL'].fmeasure) * 100.0)
    
    # return the average score
    return sum(scores) / len(scores)