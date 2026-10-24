I built Trajecta, a simulation-first robotics project for running and evaluating Panda pick-and-place experiments in MuJoCo.

The current demo combines an IK-backed end-effector controller with a task state machine that verifies bilateral grasp contact, lift, release, and placement. Each episode records its seed, outcome, task phases, and measured placement error.

In a five-episode smoke run on the bundled scene, all five trials completed successfully with a mean horizontal placement error of 7.59 mm. This is a repeated run on one fixed initial condition; the current seeds do not randomize object or goal positions, so I am treating it as a functional check rather than a robustness benchmark.

Next I plan to add randomized held-out evaluations and transition-level recording before evaluating calibration or learned policies.

Project: https://github.com/monithsurya/Trajecta
