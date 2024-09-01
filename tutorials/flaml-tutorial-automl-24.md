# AutoML 2024 - Automated Machine Learning & Tuning with FLAML in Microsoft Fabric

## Session Information

**Date and Time**: 09.09.2024, 15:30-17:00

Location:  Sorbonne University, 4 place Jussieu, 75005 Paris

Duration: 1.5 hours

For the most up-to-date information, see the [AutoML 2024 Agenda](https://2024.automl.cc/?page_id=1401) and the [tutorial page](https://2024.automl.cc/?page_id=1643).

## Abstract

In this tutorial, we will provide an in-depth and hands-on guidance on Automated Machine Learning & Tuning with FLAML in Microsoft Fabric. FLAML is a fast python library for AutoML and tuning. Microsoft Fabric is an end-to-end analytics and data platform designed for enterprises that require a unified solution. In Fabric, data scientists can use flaml.AutoML to automate their machine learning tasks. We will start with an overview of the AutoML problem and our solution. We will then introduce the hyperparameter optimization methods and 60+ estimators empowering the strong performance of FLAML. We will also demonstrate how to make the best use of FLAML in Microsoft Fabric to perform automated machine learning and hyperparameter tuning in various applications with the help of rich customization choices, parallel training and advanced auto logging functionalities. At last, we will share several new features of our solution based on our latest research and development work around FLAML in Microsoft Fabric and close the tutorial with open problems and challenges learned from AutoML practice.

## Motivation & Outline

As data becomes increasingly complex and voluminous, the demand for robust, scalable, and user-friendly tools for model selection, hyperparameter tuning, and performance optimization has never been higher. FLAML, a fast Python library for AutoML, and Microsoft Fabric, an advanced data platform, address these needs by offering a comprehensive suite of built-in machine learning tools. What sets FLAML in Microsoft Fabric apart is its unique support for visualization, auto-featurization, advanced auto logging capabilities, and a wider range of Spark models, distinguishing it from the open-source version of FLAML. Attendees of the AutoML conference will gain invaluable insights into leveraging these technologies to streamline their workflows, improve model accuracy, and enhance productivity. By mastering the integration of FLAML with Microsoft Fabric, participants can significantly reduce the time and expertise required for machine learning tasks, making this tutorial highly relevant and essential for advancing their work in data science and analytics.

In this tutorial, we will provide an in-depth and hands-on guidance on Automated Machine Learning & Tuning with FLAML in [Microsoft Fabric](https://aka.ms/fabric). FLAML (by [Wang et al., 2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/1ccc3bfa05cb37b917068778f3c4523a-Paper.pdf)) is a fast python library for AutoML and tuning. It started as a research project in Microsoft Research and has grown to a popular open-source library. It has accumulated over 3.7k stars and 4M+ downloads since its first release in December 2020. FLAML is notable for being fast, economical, and easy to customize. FLAML enhances the efficiency and productivity of machine learning and data science professionals, while delivering superior predictive performance in models. FLAML’s flexibility and customizability make it an invaluable tool for research and development. Microsoft Fabric is a comprehensive analytics and data platform designed for enterprises seeking a unified solutionIt provides data science capabilities that enable users to manage the entire data science workflow—from data exploration and cleaning, through experimentation and modeling, to model scoring and delivering predictive insights into BI reports. On Microsoft Fabric, users accelerate their model training workflows through the code-first FLAML APIs available through Fabric Notebooks. Microsoft Fabric supports tracking machine learning lifecycle with MLflow. FLAML experiments and runs could be automatically logged for you to visualize, compare and analyze. All the 60+ [models](https://learn.microsoft.com/en-us/fabric/data-science/automated-machine-learning-fabric/#supported-models) trained with flaml.AutoML will be automatically recognized and logged for further usage. We will give a hands-on tutorial on (1) how to use FLAML in Microsoft Fabric to automate typical machine learning tasks and generic tuning on user-defined functions; (2) how to make the best use of FLAML in Microsoft Fabric to perform AutoML and tuning in various applications with the help of rich customization choices, parallel training and advanced auto logging functionalities; and (3) several new features of FLAML based on our latest research and development work around FLAML in Microsoft Fabric.

Part 1. Overview of AutoML in Microsoft Fabric

- Background of AutoML & Hyperparameter tuning
- Quick introduction to FLAML and Microsoft Fabric
- Task-oriented AutoML
- Tuning generic user-defined functions

Part 2. A deep dive into FLAML in Microsoft Fabric

- Parallel training with spark and customizing estimator and metric
- Track and analyze experiments and models with auto logging

Part 3. New features on FLAML in Microsoft Fabric

- Auto Featurization
- Visualization
- Tuning in-context-learning for LLM models
