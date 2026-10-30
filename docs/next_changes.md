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
