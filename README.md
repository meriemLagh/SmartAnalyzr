### Project Overview

This project involves the design and development of a versatile and interactive application for data analysis, incorporating various machine learning algorithms. Designed to meet the growing demands for data management, exploration, and visualization, the application serves as an intuitive and accessible tool aimed at democratizing the use of data science for a wide audience.

The application primarily targets three user groups. First, students and researchers in the fields of statistics, artificial intelligence, and data science, who can use it for their academic or experimental work. Second, professionals, including data analysts and business decision-makers, who seek simple and effective ways to explore data for strategic decision-making. Lastly, the application is also designed for technology and programming enthusiasts who wish to learn or implement machine learning models in a simplified environment.

The main objective of this application is to provide an elegant and intuitive user interface that allows users to load, visualize, and analyze a variety of datasets. Through the integration of several supervised and unsupervised learning algorithms, such as regression, decision trees, random forests, support vector machines (SVM), and k-means clustering, users can not only process their data but also build and evaluate predictive models with ease. The application also includes advanced features, such as model validation, variable selection, and interactive result visualization, to enrich the user experience.

This project aims to provide a powerful yet accessible professional tool that encourages data exploration and the adoption of artificial intelligence technologies. By simplifying complex tasks like data preparation, predictive model creation, and result interpretation, the application aims to transform raw data into actionable insights, supporting decision-making and innovation across various industries.

### Application Structure

This application consists of several pages, each responsible for a specific task, which are called by the main page to execute them. Below are the different pages that make up our project:

- **Home.py**: This is the main file responsible for displaying the application when executed. It calls the functions from various files to build the application.
- **ImportDataset.py**: This file is responsible for downloading the dataset.
- **CreationDataset.py**: This file handles the creation and downloading of the dataset.
- **ExplorationData.py**: This file is responsible for displaying general information about the studied dataset.
- **NettoyageDataset.py**: This file cleans the studied dataset and downloads the cleaned version.
- **Classification.py**: This file is in charge of creating the supervised classification algorithms.
- **Regression.py**: This file creates the supervised regression algorithms.
- **Clustering.py**: This file is responsible for creating the unsupervised clustering algorithms.
- **NeuralNetworks.py**: This file is responsible for creating the deep learning algorithms.
