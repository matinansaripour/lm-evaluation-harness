from typing import List, Tuple, Dict, Any, Union

def longestCommonSubstr(s1, s2):
    m = len(s1)
    n = len(s2)

    prev = [0] * (n + 1)
    
    res = 0
    for i in range(1, m + 1):
      
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                res = max(res, curr[j])
            else:
                curr[j] = 0
        
        prev = curr
    return float(res)

def lccs(items):
    scores = []
    for pred, ref in items:
        pred = pred.strip()
        ref = ref.strip()
        pred = " ".join(pred.split())
        ref = " ".join(ref.split())
        scores.append(longestCommonSubstr(pred, ref))
    
    return scores