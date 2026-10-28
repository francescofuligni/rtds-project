# Sources for the RTDS Report

The following sources support the current project and its related-work discussion;
references use the ICLR author-year format and match [the report bibliography](../../report/references.bib).

- Barry Becker and Ronny Kohavi. **Adult**. UCI Machine Learning Repository, 1996. Dataset. URL: [https://doi.org/10.24432/C5XW20](https://doi.org/10.24432/C5XW20).

  This is the original source of the Adult data used to construct the project's private samples and disjoint public/holdout samples.

- Travis Dick, Cynthia Dwork, Michael Kearns, Terrance Liu, Aaron Roth, Giuseppe Vietri, and Zhiwei Steven Wu. **Confidence-Ranked Reconstruction of Census Microdata from Published Statistics**. *Proceedings of the National Academy of Sciences*, 120(8):e2218605120, 2023. URL: [https://doi.org/10.1073/pnas.2218605120](https://doi.org/10.1073/pnas.2218605120).

  This paper motivates ranking reconstructed records by confidence and comparing recovery against informed public-data baselines, without implying that our implementation reproduces its attack.

- Julien Ferry, Ricardo Fukasawa, Timothée Pascal, and Thibaut Vidal. **Trained Random Forests Completely Reveal your Dataset**. In *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of Proceedings of Machine Learning Research, pp. 13545–13569. PMLR, 2024. URL: [https://proceedings.mlr.press/v235/ferry24a.html](https://proceedings.mlr.press/v235/ferry24a.html).

  This paper introduces DRAFT, whose path and leaf-count reconstruction formulation is adapted to the project's non-bootstrap TreeFARMS releases and known per-row labels.

- Sture Holm. **A Simple Sequentially Rejective Multiple Test Procedure**. *Scandinavian Journal of Statistics*, 6(2):65–70, 1979. URL: [https://www.jstor.org/stable/4615733](https://www.jstor.org/stable/4615733).

  This paper defines the multiple-testing correction implemented for the main experiment's per-release-size paired comparisons.

- Ethan Hsu, Tony Cao, Lesia Semenova, and Chudi Zhong. **The Rashomon Set Has It All: Analyzing Trustworthiness of Trees under Multiplicity**. In *Advances in Neural Information Processing Systems*, volume 38, Datasets and Benchmarks Track. Curran Associates, Inc., 2025. URL: [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c914b154c323d4d6c681d09dd679438c-Abstract-Datasets_and_Benchmarks_Track.html).

  This paper provides the broader benchmarking context for studying trustworthiness across sparse-tree Rashomon sets; our project focuses on reconstruction privacy rather than reproducing its full benchmark.

- Ethan Hsu, Harry Chen, Chudi Zhong, and Lesia Semenova. **The Double-Edged Nature of the Rashomon Set for Trustworthy Machine Learning**. In *Proceedings of the 43rd International Conference on Machine Learning*, volume 306 of Proceedings of Machine Learning Research, pp. 44452–44490. PMLR, 2026. URL: [https://proceedings.mlr.press/v306/hsu26a.html](https://proceedings.mlr.press/v306/hsu26a.html).

  This paper establishes the privacy motivation for examining how releasing multiple near-optimal trees exposes information about their shared training data.

- Rui Xin, Chudi Zhong, Zhi Chen, Takuya Takagi, Margo Seltzer, and Cynthia Rudin. **Exploring the Whole Rashomon Set of Sparse Decision Trees**. In *Advances in Neural Information Processing Systems*, volume 35, pp. 14071–14084. Curran Associates, Inc., 2022. URL: [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5afaa8b4dd18eb1eed055d2d821b58ae-Abstract-Conference.html).

  This paper introduces TreeFARMS, the algorithm used to enumerate and sample the near-optimal decision trees released in our experiments.
