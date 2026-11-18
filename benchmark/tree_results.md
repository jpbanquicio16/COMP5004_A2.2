# Tree benchmarks

Python 3.12.13, NumPy 1.26.4, best of 3 runs.

## (a) Loop vs vectorised (max_depth=5; 20 yes/no features or 10 Gaussian features, 3 classes; predict on 20,000 rows)

| data | n_train | fit loop (ms) | fit vec (ms) | fit speed-up | predict loop (ms) | predict vec (ms) | predict speed-up | identical tree + predictions |
|---|---|---|---|---|---|---|---|---|
| 0/1 | 200 | 32.5 | 5.2 | 6.2x | 105.1 | 2.1 | 49.7x | yes (100%) |
| 0/1 | 1000 | 134.5 | 24.0 | 5.6x | 115.6 | 2.1 | 55.8x | yes (100%) |
| 0/1 | 5000 | 655.2 | 114.6 | 5.7x | 117.9 | 2.2 | 53.7x | yes (100%) |
| continuous | 200 | 178.9 | 4.4 | 41.1x | 107.8 | 1.9 | 58.2x | yes (100%) |
| continuous | 1000 | 978.6 | 17.4 | 56.2x | 115.0 | 1.8 | 62.7x | yes (100%) |
| continuous | 3000 | 2847.4 | 46.9 | 60.8x | 115.5 | 2.0 | 58.6x | yes (100%) |

## (b) buffer_size sweep (20,000-row stream, 50-row chunks, label rule changes at row 10,000; mean of 5 seeds)

| buffer_size | retrains | total retrain time (s) | time / retrain (ms) | prequential accuracy | peak model memory (KB) | rows to recover after change |
|---|---|---|---|---|---|---|
| 50 | 400 | 0.544 | 1.4 | 0.796 | 1.6 | 60 |
| 100 | 200 | 0.545 | 2.7 | 0.818 | 10.3 | 110 |
| 250 | 80 | 0.534 | 6.7 | 0.855 | 35.3 | 350 |
| 500 | 40 | 0.535 | 13.4 | 0.898 | 76.6 | 550 |
| 1000 | 20 | 0.533 | 26.6 | 0.892 | 158.8 | 1000 |
| 2500 | 8 | 0.487 | 60.9 | 0.844 | 405.0 | 2500 |
