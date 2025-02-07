# metalearning

this is a project exploring the identification and amplification of core learning features in rl agents. the goal is to extract minimal, generative primitives that capture fundamental learning dynamics—no more task-specific hacks.

through an apd-driven pipeline $^{[1]}$ (see `@scripts` for the detailed steps), i run a gridworld transformer training (via `apd_test_on_transformer.py`) to dump model snapshots every epoch. then, a follow-up analysis (using `apd_train.py`) leverages custom cuda kernels (speed is EVERYTHING, bc performance is key) to dissect layer-wise weight dynamics. the resulting data gets fed into a jupyter notebook where i can easily visualize learning waves and feature evolution.

the ipynb is a little new, haven't added anything meaningful yet.

## current setup

- rn: run transformer training and snapshot generation (apd test) from the `@scripts` folder.
- then, run the apd analysis (apd train) also in `@scripts`.
- finally, analyze the ipynb to extract **CORE** and **ENVIRONMENT-INVARIANT** features.

## future directions

- add backward passes
- later: write more sophisticated kernels (or get one online) to further speed up apd computations.
- eventually: expand to cross-environment tests, adjust gridworld parameters, and experiment with interventions on these learning primitives.
- i originally planned on using saes (see `project.tex`) for feature extraction—problems arose, so i pivoted to the direct apd approach for cleaner insight.

## references

[1] Dan Braun et al., _Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition_ (2025). [https://arxiv.org/abs/2501.14926](https://arxiv.org/abs/2501.14926)
