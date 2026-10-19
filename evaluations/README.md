# Evaluations

All numbers use the competition metric: the mean absolute per-step reward over a scenario, averaged over scenarios (lower is better).

## Tables (CSV, open in Excel)

| File | Content |
|---|---|
| `official_summary.csv` | Official score of every controller over the 30 shipped scenarios, 30-day episodes |
| `official_per_scenario.csv` | Same evaluation per scenario: DPC, the four baselines, best baseline, improvement and the DPC comfort, energy and carbon terms |
| `loop_benchmark_6_scenarios_2_days.csv` | Each DPC version on the internal benchmark (6 scenarios, 2 days each), with the world model it was trained on |
| `world_model_accuracy.csv` | World-model temperature error versus prediction horizon on unseen buildings, against persistence |
| `official_eval_dpc_v3.txt` | Raw log of the official evaluation run |
| `timeseries_*.csv` | Step-by-step trajectories of DPC and PID With Water Reset on two scenarios |

## Figures (`figures/`, PNG and PDF)

1. `fig1_official_scores` official score of every controller.
2. `fig2_improvement_heatmap` improvement over PID With Water Reset for every building and climate.
3. `fig3_per_scenario` per-scenario score of DPC and PID With Water Reset.
4. `fig4_loop_progress` policy improvement across the model-based loop.
5. `fig5_world_model_accuracy` world-model error versus horizon.
6. `fig6_timeseries_phoenix`, `fig7_timeseries_fairbanks` three days under both controllers.

## Regenerating

From the repository root:

```
python -m sbwm.report_tables
python -m sbwm.record_timeseries --scenario building_train_8/phoenix_4 --weights policies/dpc_v3.pt
python -m sbwm.make_figures
```
