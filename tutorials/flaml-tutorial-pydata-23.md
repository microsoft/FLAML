# PyData Seattle 2023 - Automated Machine Learning & Tuning with FLAML

## Session Information

**Date and Time**: 04-26, 09:00–10:30 PT.

Location:

Duration: 1.5 hours

For the most up-to-date information, see the [PyData Seattle 2023 Agenda](https://seattle2023.pydata.org/cfp/talk/BYRA8H/)

## Lab Forum Slides (To Be Added)

## What Will You Learn?


In this session, we will provide an in-depth and hands-on tutorial on Automated Machine Learning & Tuning with a fast python library named FLAML. We will start with an overview of the AutoML problem and the FLAML library. We will then introduce the hyperparameter optimization methods empowering the strong performance of FLAML. We will also demonstrate how to make the best use of FLAML to perform automated machine learning and hyperparameter tuning in various applications with the help of rich customization choices and advanced functionalities provided by FLAML. At last, we will share several new features of the library based on our latest research and development work around FLAML and close the tutorial with open problems and challenges learned from AutoML practice.

In this session, we will provide an in-depth and hands-on tutorial on Automated Machine Learning & Tuning with a fast python library FLAML. AutoML is playing an increasingly impactful role in modern machine learning and artificial intelligence lifecycles. According to a recent report from ReportLinker, the AutoML market is predicted to increase from $346.2 million in 2020 to $14,830.8 million in 2030, demonstrating a compound annual growth rate of 45.6%[1].

FLAML[2] started as a research project in Microsoft Research, and has grown to a popular open-source library. It has accumulated over 2k stars and about 800k downloads since its first release in December 2020. Compared to other AutoML tools, FLAML is notable for being fast, economical, and easy to customize. FLAML can increase the productivity of ML and data science practitioners while getting superior model prediction performance. The flexibility and customizability make it a powerful tool for R&D.

In this session, we will give a hands-on tutorial on (1) how to use FLAML to automate typical machine learning tasks and generic tuning on user-defined functions; (2) how to make the best use of FLAML to perform AutoML and tuning in various applications with the help of rich customization choices and advanced functionalities provided by FLAML; and (3) several new features of FLAML based on our latest research and development work around FLAML.

Target audience and prerequisites. The primary target audience of this tutorial is machine learning and data science practitioners, including data scientists, machine learning engineers, domain experts, and researchers, especially researchers who are interested in the automation of machine learning or artificial intelligence methods. The audiences are assumed to have the basic knowledge of machine learning, data science, and Python programming language.

Strategies and needs of the hands-on tutorial. This tutorial is intended to be hands-on and interactive. Most of the topics covered in the tutorial will be accompanied by demonstrations implemented in Jupyter notebooks.

### Tutorial Outline
**Part 1. Overview of AutoML and FLAML:** We will introduce the background of automated machine learning (AutoML) & automated hyperparameter tuning. We will then introduce a fast and lightweight python library for automated machine learning and general hyperparameter tuning. We will brief the overall design, innovations, target users, and software impact. We will demo the following use cases:
- How to use FLAML to do task-oriented AutoML, where tasks include typical machine learning tasks, such as regression and classification;
- How to use FLAML to tune generic user-defined functions. Here the user-defined functions include workloads beyond machine learning model training.

**Part 2: A deep dive into FLAML**: In this section, we will have a deep dive of FLAML form the following aspects:
- The hyperparameter optimization and automated machine learning methods[2-4] that empower FLAML's strong performance;
- Case studies to illustrate how to make the best use of FLAML with the rich customization choices and advanced functionalities, including zero-shot AutoML[5];
- How to use FLAML in different platforms, such as Spark, MLFlow, Azure, Ray, etc.

**Part 3: New features on FLAML**: In this section, we will introduce cutting-edge research and development around FLAML. More specifically, we will briefly introduce and demonstrate how FLAML can be useful in performing the following tasks/scenarios,
- AutoML for time series forecasting tasks, covering both single- and multi-dimensional time series data.
- AutoML for NLP tasks[6], including integration with Huggingface transformers like BERT and OpenAI models like GPT-3.
- Targeted HPO with constraints or multiple objectives[7, 8], which commonly exist in real-world deployments.

At last, we will share open problems, and challenges learned from AutoML practice.

Accompanying materials: Most of the topics covered in the tutorial will be accompanied by demonstrations implemented in Jupyter notebooks. The audience can access the notebooks and run the code examples directly in Colab. Lab components will be publicly available on Github.

**References**

[1] 2021. AutoML Market. ReportLinker (Nov 2021). https://www.reportlinker.com/p06191010/AutoML-Market.html?utm_source=GNW

[2] Chi Wang, Qingyun Wu, Markus Weimer, and Erkang Zhu. 2021. FLAML: A Fast and Lightweight AutoML Library. In MLSys.

[3] Chi Wang, Qingyun Wu, Silu Huang, and Amin Saied. 2021. Economic hyperparameter optimization with blended search strategy. In International Conference on Learning Representations.

[4] Qingyun Wu, Chi Wang, and Silu Huang. 2021. Frugal optimization for cost-related hyperparameters. In Proceedings of the AAAI Conference on Artificial Intelligence.

[5] Moe Kayali and Chi Wang. 2022. Mining Robust Default Configurations for Resource-constrained AutoML. https://doi.org/10.48550/ARXIV.2202.09927
[6] Xueqing Liu and Chi Wang. 2021. An Empirical Study on Hyperparameter Optimization for Fine-Tuning Pre-trained Language Models. In the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing.

[7] Yi-Wei Chen, Chi Wang, Amin Saied, and Rui Zhuang. 2022. ACE: Adaptive Constraint-aware Early Stopping in Hyperparameter Optimization. arXiv preprint arXiv:2208.02922 (2022).

[8] Shaokun Zhang, Feiran Jia, Chi Wang, Qingyun Wu. 2022. Targeted Hyperparameter Optimization with Lexicographic Preferences Over Multiple Objectives. ICLR 2023 (notable-top-5%).
