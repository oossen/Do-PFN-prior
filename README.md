## About

This repository contains code for a prior for a regression prior-fitted network [1]. The generated prior data is similar to that of the prior developed for Do-PFN [2], though the code is structured quite differently, incorporating many elements of an expanded version of the prior by Arik Reuter (private communication).

## Features
- an SCM-based prior, with implementation closely following the specification of an SCM by structural equations
- simple code base that should make it relatively easy to adapt things as needed
- easy configuration of basic parameters with configuration file
- compatible out of the box with the NanoTabPFN training loop from TFM-Playground [3]
- fully seeded: on-the-fly data generation is completely determined by one integer seed
- supports computing the likelihood of a target `y` given features `x` and a fixed SCM, and thus training with "soft labels"/cross-entropy loss

## Usage
The script `example.py` contains code for pretraining a NanoTabPFN model on the prior.

### References
1. **Müller, S.**, **Hollmann, N.**, **Pineda Arango, S.**, **Grabocka, J.**, **Hutter, F.** *Transformers Can Do Bayesian Inference*. arXiv preprint arXiv:2112.10510, 2021. [https://arxiv.org/abs/2112.10510](https://arxiv.org/abs/2112.10510)  
2. **Robertson, J.**, **Reuter, A.**, **Guo, S.**, **Hollmann, N.**, **Hutter, F.**, **Schölkopf, B.** *Do-PFN: In-Context Learning for Causal Effect Estimation*. arXiv preprint arXiv:2506.06039, 2025. [https://arxiv.org/abs/2506.06039](https://arxiv.org/abs/2506.06039)  
05564, 2025. [https://arxiv.org/abs/2502.05564](https://arxiv.org/abs/2502.05564)  
3. **Automl**. *TFM-Playground*. GitHub repository, 2024. Available at: [https://github.com/automl/TFM-Playground](https://github.com/automl/TFM-Playground)
