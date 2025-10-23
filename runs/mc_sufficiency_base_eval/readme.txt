This run uses calibration (T*) **only** for ranking/refinement; final overlays/metrics are BASE SAM.

- Sufficiency scores s_j and the greedy 'smallest sufficient set' are computed under T*.
- The side-by-side figures and the reported mIoU/Dice in titles are computed on BASE:
    p = sigmoid(z), threshold = 0.5, candidate chosen by lowest CE at T=1.
- perfile/*_summary_metrics.csv records both BASE and T* metrics for transparency.

Prompt dots: green = positive (inclusion), red = negative (exclusion), 6 px diameter.

All CEs use natural logs (nats).
