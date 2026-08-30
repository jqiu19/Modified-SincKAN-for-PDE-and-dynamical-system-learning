| Dimension | MLP | Modified MLP | KAN | SincKAN | Modified SincKAN |
| --- | ---: | ---: | ---: | ---: | ---: |
| d = 2 | 6.5484e-03 | 3.3124e-05 | 1.7148e-03 | 4.1277e+00 | 1.5205e-04 |
| d = 10 | 6.9039e-02 | 2.8347e-03 | 4.8457e-03 | 4.2641e+00 | 1.5782e-03 |
| d = 20 |  |  |  |  |  |
| d = 50 |  |  |  |  |  |

Notes:
- Entries are three-seed mean final RMSE values.
- `d = 2` uses the current formal comparison table in `steady_poisson_dim2_final_compare.md`.
- `d = 10` uses the current formal runs, with `Modified SincKAN` from the new `32/8 + separate_residual + aug_scale=0.1` configuration.
