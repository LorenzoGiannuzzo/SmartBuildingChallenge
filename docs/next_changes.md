# Changes planned for the next round

Findings from the control figures of v10_base_f03 (results/inspect), to be
implemented together with the next model change.

1. Non-working days. On holidays (for example Friday 10 November 2023,
   Denver 8) the comfort-mode flag of the observation stays on, but the zone
   setpoints stay in the reduced band and nobody arrives; every controller
   (including pid_water) heats as on a working day, about 1100 kWh of gas,
   17% of that week. Plan: when the setpoints stay in the reduced band at the
   scheduled comfort start, the schedule features treat the day as a weekend,
   in training and at deployment alike.
2. Coldest zones. In cold weather the worst zones stay 2-3 K below the band
   all day; v10_base_f03 opens their reheat 55% on average (A10 79%,
   pid_water 97%) and keeps the supply water at its 37 C minimum. Simulator
   test on Denver 8 (7 days), v10_base_f03 with a floor on the water command:
   no floor 0.0174 (comfort 0.0121, energy and carbon 0.0053); floor -0.5
   0.0160; floor 0.0 (mid range) 0.0155, comfort -15% at the same energy.
   Earlier test on A10 at Fairbanks 1 with the water at its maximum: -3%.
   Next: evaluate a water floor on all 30 official scenarios (as a checkpoint
   option, like the command filter), and check why the world model does not
   value a warmer water supply.
3. Cold nights and weekends. Worst zones down to 8-9 C on cold nights (-9 C
   at Fairbanks over the weekend), with a 1000 kW gas peak at the morning
   start. Addressed by the long horizon (v11_long_fit3).
4. Residual slow oscillation of gas and reheat (period about one hour),
   much smaller than without the filter.

## Planned for round v14 (decided 3 November, pending the v13 results)

5. 48-hour configuration. If v13_mem_long48 improves the Monday start-up
   without degrading the rest:
   - world model trained on unrolls of 4, 12, 24 and 48 hours (lengths
     48 144 288 576), with gradient damping in the unroll if needed;
   - DPC horizon 24 to 48 hours on that world model;
   - start states drawn more often on Friday and Saturday, so that more
     rollouts contain a weekend and the following Monday;
   - memory (1, 6, 24 h) in the policy; a 72 h average to test later;
   - frost-protection floor at deployment (about 10 C) as a safety net,
     kept even if the policy learns not to let the building freeze.
   If the 48 h policy on the 24 h world model degrades instead, the
   world-model accuracy beyond 24 h is the bottleneck and the 48 h world
   model becomes the first priority.

## Round v15: physics-based zone features (implemented, queued after v14)

- `sbwm/physics.py`: per-zone state updated at every decision, identical in
  the DPC rollouts and at deployment (the logged prefix of a rollout is
  replayed with the same recursion, in numpy for speed); 8 extra zone inputs
  with `--phys`: indoor-outdoor difference, temperature change over 30 min,
  1 h and 4 h (on the temperature smoothed over 15 min against the sensor
  noise of 0.3-0.6 K), estimated time constant of the free-floating zone and
  recovery rate, free-floating temperature expected at the next comfort start
  and preheat slack (the optimal-start signal).
- New inputs start with zero weights (`init_policy`), so each policy begins as
  transformer_v6.
- `scripts/server_round_v15.sh`: both v14 recipes (2wm 24 h, 48 h on WM v14)
  with `--phys`, three seeds each, ranked with round v14.
- `sbwm/make_bundle.py` builds the Kaggle zip (policy.py now imports
  physics.py, so bundles must include it).
- Open: training rollouts use noise-free temperatures, deployment noisy ones
  (smoothed); sensor noise could be injected in training if the 30-min change
  turns out to be unreliable.
