# Vanilla speculative decoding smoke test — RTX 4090, LLaMA-3-8B + LLaMA-3.2-1B

Baseline: 52.80 tok/s

Prompt | tokens | acc rate | tok/verify | tok/s
-------|--------|----------|------------|------
photosynthesis | 169 | 91.5% | 4.33 | 70.9
fibonacci      | 259 | 86.0% | 3.75 | 71.0
lighthouse     | 256 | 75.3% | 3.12 | 59.8
France capital | 256 | 89.3% | 4.00 | 77.7
seasons poem   | 258 | 77.5% | 3.15 | 61.2

Median: 70.94 tok/s
Median acceptance: 86.0%
Speedup vs baseline: 1.34x
