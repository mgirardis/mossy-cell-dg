python -OO plot_lifdt_param_search.py -paramset EIFDTBoundSigKLR -DeltaGK 0.0 -I0 0.07:0.3:24 -avg all -avgmodel same -savedata
python -OO run_calc_theta0_I_curve.py -paramset EIFDTBoundSigKLR -DeltaGK 0.0 -I0 0.07:0.3:24 -savedata
python -OO run_threshold_decay_experiments.py -paramset EIFDTBoundSigKLR -DeltaGK 0.0 -Iprobe 0.35 -Istim 0.30 -ntrials 10 -savedata
python -OO run_current_injection.py -paramset EIFDTBoundSigKLR -DeltaGK 0.0 -I0 0.07:0.35:29 -T 1000 -tStim 10 -save -savevoltage