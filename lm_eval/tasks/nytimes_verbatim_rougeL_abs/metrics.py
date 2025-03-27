

def _score_lcs(target_tokens, prediction_tokens):
  """Computes LCS (Longest Common Subsequence) rouge scores.

  Args:
    target_tokens: Tokens from the target text.
    prediction_tokens: Tokens from the predicted text.
  Returns:
    A Score object containing computed scores.
  """

  # Compute length of LCS from the bottom up in a table (DP appproach).
  lcs_table = _lcs_table(target_tokens, prediction_tokens)
  lcs_length = lcs_table[-1][-1]

  return lcs_length


def _lcs_table(ref, can):
  """Create 2-d LCS score table."""
  rows = len(ref)
  cols = len(can)
  lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
  for i in range(1, rows + 1):
    for j in range(1, cols + 1):
      if ref[i - 1] == can[j - 1]:
        lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
      else:
        lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
  return lcs_table


def rougeL_abs(references, predictions, **kwargs):
    scores = []
    for pred, ref in zip(predictions, references):
        pred = pred.strip()
        ref = ref.strip()
        pred = " ".join(pred.split())
        ref = " ".join(ref.split())
        scores.append(float(_score_lcs(ref, pred)))

    # return the average score
    return sum(scores) / len(scores)