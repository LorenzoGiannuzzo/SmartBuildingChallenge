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
  replayed with the same recursion, in numpy for speed); extra zone inputs
  with `--phys` (14 in total): indoor-outdoor difference, heating and cooling
  setpoint minus zone temperature and comfort setpoint minus outdoor air
  (setpoint-referenced, after Brandi et al. 2020), temperature change over 30 min,
  1 h and 4 h (on the temperature smoothed over 15 min against the sensor
  noise of 0.3-0.6 K), and over 12, 24 and 48 h (hourly samples), estimated time constant of the free-floating zone and
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
- Light frost guard (5-8 C, `sbwm.set_frost src dst 5 8`) kept on every
  policy from round v15 on, including the Kaggle bundles, so the building never
  drops below about 5 C; on rounds v11/v13 it costs about 0.0003 on the
  14-day score (0.0106 to 0.0109).
- Slower decisions: `action_hold` in the checkpoint (`sbwm.set_hold src dst k`)
  repeats the commands for k decisions at deployment; round v15 evaluates
  10 and 15 min on its best policy. If it helps, the next step is a hierarchy
  inside training (plant commands every 15-30 min, zones every 5 min with the
  filter); the old smoothness penalty in training made things worse.

## Round v16 (queued after v15)

- Kappa (comfort weight of the DPC objective) 1.0 and 1.4 against 1.8 on the
  v14_new2wm recipe, two seeds each, plus a second seed at 1.8. Evaluation is
  always with the official reward; world model v14 overestimates discomfort
  (comfort ratio 1.22 replay, 1.08 closed loop), so 1.8 may double-count it.
  History: on WM-1, kappa 1.0 beat 1.8 (abl_A10 0.0137 vs abl_A 0.0151, 7 d);
  on WM-3, 1.8 beat 1.0 (v10_base 0.0156 vs v10_base10 0.0299, 7 d).
- Minimum on/off time of boiler loop and air handlers at deployment
  (`min_run`, anti-short-cycling). On Fairbanks 1 (14 d) with v14_48_s0, 30 min
  cut the weekend boiler switches from 232 to 25 and improved the score from
  0.0659 to 0.0650.
- Plant commands (air handlers, hot-water loop) held for 15 or 30 min at
  deployment (`plant_hold`), zones still every 5 min; tested in round v16 on
  the best policy. Round v15 holds all commands (`action_hold`).
- `sbwm/set_option.py` sets any deployment option of a checkpoint.

## Oracle bound (queued after v16)

- `sbwm/oracle.py`: receding-horizon search with the true simulator. Every
  hour, 31 candidate modifications of the policy commands (plant forced
  off/on, supply-water, supply-air and reheat offsets) are tried on forked
  copies of the simulator over 24 h, the policy continuing after the first
  hour; the best is applied. The pure policy is always a candidate.
  Simulator cloning works by sharing the constant sparse LU factorization of
  the thermal solver (deep copy verified identical on Denver 8).
- `scripts/server_oracle.sh`: best policy of v14-v16 alone and in the oracle,
  7 days on Fairbanks 1, Phoenix 5 and Denver 8.
- Next, depending on the gap: planning at deployment with the world model
  (gradient refinement of the policy commands every hour), then online
  correction of the world model per building.

## Round wm15: structured power (scripts/server_round_wm15.sh)

- `sbwm/components.py` reads, during collection, the power of every device:
  zone reheat coils, AHU fans and cooling, boiler tank losses (the tank term is
  the rest of the gas). Checked on Fairbanks 1: AHU parts sum to the reward
  totals exactly, and in steady heating the gas equals the sum of the coils.
- World model `--structured`: zone tokens predict their coil heat, AHU tokens
  fans and cooling, the HWS token pump, losses and tank; gas and electricity
  are the sums. Component losses in the units of the totals, plus
  `--w-hourly` (error of the hourly mean power). World model v14, started as a
  structured model, puts most of the gas in the HWS standby term (coils 0.12 of
  the true heat, losses 200 times): the totals were right for the wrong reason.
- Diagnosis figure `sbwm/wm_error_plots.py`: hourly power profiles and
  component ratios (pred/true) next to the temperature error.
- Round: collection (on-policy v14_new2wm with probes, Fairbanks folders,
  clean diagnosis runs), WM v15 from v14, diagnosis and figures for v14/v15,
  policies on v15 (24 h with v14, 48 h), evaluation, control figures.
