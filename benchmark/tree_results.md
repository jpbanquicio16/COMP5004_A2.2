# Tree benchmarks

## (a) Loop vs vectorised (max_depth=5, 20 yes/no features)

| n_train | fit loop (s) | fit vec (s) | fit speed-up | predict loop (s) | predict vec (s) | predict speed-up | prediction agreement |
|---|---|---|---|---|---|---|---|
| 200 | 0.0068 | 0.0048 | 1.4x | 0.0066 | 0.00108 | 6.1x | 1.000 |
| 1000 | 0.0312 | 0.0105 | 3.0x | 0.0074 | 0.00124 | 5.9x | 1.000 |
| 5000 | 0.1161 | 0.0151 | 7.7x | 0.0066 | 0.00130 | 5.1x | 1.000 |

## (b) buffer_size sweep (20,000-sample stream, 50-row chunks, concept flips at the midpoint)

| buffer_size | retrains | total retrain time (s) | time / retrain (s) | prequential accuracy |
|---|---|---|---|---|
| 50 | 400 | 0.7065 | 0.00177 | 0.795 |
| 100 | 200 | 0.5695 | 0.00285 | 0.815 |
| 250 | 80 | 0.3669 | 0.00459 | 0.851 |
| 500 | 40 | 0.2378 | 0.00595 | 0.900 |
| 1000 | 20 | 0.1537 | 0.00768 | 0.892 |
| 2500 | 8 | 0.0812 | 0.01015 | 0.844 |
