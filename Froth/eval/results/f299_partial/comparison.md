| method | found / GT | recall (95% CI) | mean IoU | diam ratio | D32 ratio (matched) | missed / touched | n pred |
|---|---|---|---|---|---|---|---|
| Cellpose (zero-shot) | 26/26 | 1.00 (0.87-1.00) | 0.80 | 0.99 | 0.97 | 0 / 0 | 341 |
| BubSAM3 | 23/26 | 0.88 (0.71-0.96) | 0.71 | 0.93 | 0.91 | 0 / 3 | 597 |
| bubseg (ROI = whole image) | 13/26 | 0.50 (0.32-0.68) | 0.61 | 0.88 | 0.88 | 0 / 13 | 207 |

Partially labelled case: precision is not computable and is not shown.
