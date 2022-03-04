Runs simulations and plots data for all the models that were developed and compared for the paper

Trinh A-T, Girardi-Schappo M, Béïque J-C, Longtin A, Maler L (2022): Dentate gyrus mossy cells exhibit sparse coding via adaptive spike threshold dynamics.

# project structure

 * `modules/neurons.py` -> the classes that contain the Runge-Kutta and other numerical methods for all the models (each model is defined as a class)

 * `data/experiments` -> folder that contains the experimental data that was used to compare with the models

 * `data/simulations` -> folder that containes the data that is plotted in the `plot_eifdtboundsigklr_noKcurr_data_paper.ipynb` notebook, and is presented in the paper. This data was generated by running the simulations in `run_paper_sim.sh`

# running
to run the basic paper simulations

    ./run_paper_sim.sh

# plotting
to plot the data we have in the paper (included in the folder `data`), use the following jupyter notebooks

paper figures
    plot_eifdtboundsigklr_noKcurr_data_paper.ipynb

# get help
the other included scripts in this main directory are meant to plot different quantities of any of the considered model, get help by running

    python plot_currinj_vs_current.py --help
    python plot_currinj_vs_spk.py --help
    python plot_lifdt_param_search.py --help
    python plot_lifdt_voltage.py --help
    python run_calc_theta0_I_curve.py --help
    python run_current_injection.py --help
    python run_threshold_decay_experiments.py --help
