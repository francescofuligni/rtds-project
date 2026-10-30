# Sources for the RTDS Report

The following sources are cited in the current report and support its methods,
data, or related-work discussion. They match [the report bibliography](../../report/references.bib).

- John M. Abowd, Tamara Adams, Robert Ashmead, David Darais, Sourya Dey, Simson Garfinkel, Nathan Goldschlag, Michael B. Hawes, Daniel Kifer, Philip Leclerc, Ethan Lew, Scott Moore, Rolando A. Rodríguez, Ramy N. Tadros, and Lars Vilhuber. **A Simulated Reconstruction and Reidentification Attack on the 2010 U.S. Census**. *Harvard Data Science Review*, 7(3), 2025. URL: [https://doi.org/10.1162/99608f92.4a1ebf70](https://doi.org/10.1162/99608f92.4a1ebf70).

  This paper motivates census reconstruction and population-informed baselines.

- Barry Becker and Ronny Kohavi. **Adult**. UCI Machine Learning Repository, 1996. Dataset. URL: [https://doi.org/10.24432/C5XW20](https://doi.org/10.24432/C5XW20).

  This is the original source of the Adult data used to construct the project's private samples and disjoint public/holdout samples.

- Travis Dick, Cynthia Dwork, Michael Kearns, Terrance Liu, Aaron Roth, Giuseppe Vietri, and Zhiwei Steven Wu. **Confidence-Ranked Reconstruction of Census Microdata from Published Statistics**. *Proceedings of the National Academy of Sciences*, 120(8):e2218605120, 2023. URL: [https://doi.org/10.1073/pnas.2218605120](https://doi.org/10.1073/pnas.2218605120).

  This paper motivates ranking reconstructed records by confidence and comparing recovery against informed public-data baselines, without implying that our implementation reproduces its attack.

- Julien Ferry, Ricardo Fukasawa, Timothée Pascal, and Thibaut Vidal. **Trained Random Forests Completely Reveal your Dataset**. In *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of Proceedings of Machine Learning Research, pp. 13545–13569. PMLR, 2024. URL: [https://proceedings.mlr.press/v235/ferry24a.html](https://proceedings.mlr.press/v235/ferry24a.html).

  This paper introduces DRAFT, whose path and leaf-count reconstruction formulation is adapted to the project's non-bootstrap TreeFARMS releases and known per-row labels.

- Julien Ferry, Ulrich Aïvodji, Sébastien Gambs, Marie-José Huguet, and Mohamed Siala. **Probabilistic Dataset Reconstruction from Interpretable Models**. In *2024 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)*, pp. 1–17. IEEE, 2024. URL: [https://doi.org/10.1109/SaTML59370.2024.00009](https://doi.org/10.1109/SaTML59370.2024.00009).

  This paper studies uncertainty across datasets consistent with an interpretable model.

- Alice Gorgé, Julien Ferry, Sébastien Gambs, and Thibaut Vidal. **Training Set Reconstruction from Differentially Private Forests: How Effective is DP?** In *2026 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)*, pp. 1136–1171. IEEE, 2026. URL: [https://doi.org/10.1109/SaTML68715.2026.00067](https://doi.org/10.1109/SaTML68715.2026.00067).

  This paper extends tree reconstruction to differentially private forests.

- Sture Holm. **A Simple Sequentially Rejective Multiple Test Procedure**. *Scandinavian Journal of Statistics*, 6(2):65–70, 1979. URL: [https://www.jstor.org/stable/4615733](https://www.jstor.org/stable/4615733).

  This paper defines the multiple-testing correction implemented for the main experiment's per-release-size paired comparisons.

- Ethan Hsu, Harry Chen, Chudi Zhong, and Lesia Semenova. **The Double-Edged Nature of the Rashomon Set for Trustworthy Machine Learning**. In *Proceedings of the 43rd International Conference on Machine Learning*, volume 306 of Proceedings of Machine Learning Research, pp. 44452–44490. PMLR, 2026. URL: [https://proceedings.mlr.press/v306/hsu26a.html](https://proceedings.mlr.press/v306/hsu26a.html).

  This paper establishes the privacy motivation for examining how releasing multiple near-optimal trees exposes information about their shared training data.

- Bargav Jayaraman and David Evans. **Are Attribute Inference Attacks Just Imputation?** In *Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security*, pp. 1569–1582. Association for Computing Machinery, 2022. URL: [https://doi.org/10.1145/3548606.3560663](https://doi.org/10.1145/3548606.3560663).

  This paper motivates assessing what released models add beyond distributional knowledge; our unconditional marginal control does not reproduce its conditional imputation attacks.

- Shagufta Mehnaz, Sayanton V. Dibbo, Ehsanul Kabir, Ninghui Li, and Elisa Bertino. **Are Your Sensitive Attributes Private? Novel Model Inversion Attribute Inference Attacks on Classification Models**. In *31st USENIX Security Symposium (USENIX Security 22)*, pp. 4579–4596. USENIX Association, 2022. URL: [official proceedings](https://www.usenix.org/conference/usenixsecurity22/presentation/mehnaz).

  This paper studies attribute identity in black-box inference, a setting distinct from the project's known-column reconstruction.

- Cynthia Rudin. **Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead**. *Nature Machine Intelligence*, 1(5):206–215, 2019. URL: [https://doi.org/10.1038/s42256-019-0048-x](https://doi.org/10.1038/s42256-019-0048-x).

  This paper motivates the role of interpretable models in high-stakes decisions.

- Rui Wen, Yiyong Liu, Michael Backes, and Yang Zhang. **SoK: Data Reconstruction Attacks Against Machine Learning Models: Definition, Metrics, and Benchmark**. In *34th USENIX Security Symposium (USENIX Security 25)*, pp. 5601–5620. USENIX Association, 2025. URL: [official proceedings](https://www.usenix.org/conference/usenixsecurity25/presentation/wen).

  This paper supports distinguishing reconstruction fidelity from coverage and declaring matching assumptions, without treating our tabular target-error metric as a reproduction of its vision benchmark.

- Frank Wilcoxon. **Individual Comparisons by Ranking Methods**. *Biometrics Bulletin*, 1(6), 1945. URL: [https://doi.org/10.2307/3001968](https://doi.org/10.2307/3001968).

  This paper defines the paired signed-rank test used in the report's seed-level comparisons.

- Rui Xin, Chudi Zhong, Zhi Chen, Takuya Takagi, Margo Seltzer, and Cynthia Rudin. **Exploring the Whole Rashomon Set of Sparse Decision Trees**. In *Advances in Neural Information Processing Systems*, volume 35, pp. 14071–14084. Curran Associates, Inc., 2022. URL: [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5afaa8b4dd18eb1eed055d2d821b58ae-Abstract-Conference.html).

  This paper introduces TreeFARMS, the algorithm used to enumerate and sample the near-optimal decision trees released in our experiments.
