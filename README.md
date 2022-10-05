# Neural_Network_Charity_Analysis
Analysis of non-profit data using machine learning and neural networks to develop a binary classifier that predicts donor applicant success rate.

## Overview
Using machine learning and neural networks, this analysis utilizes a dataset provided by Alphabet Soup, a philanthropic non-profit organization, to create a binary classifier that is capable of predicting whether recipients of the organization's funding will be successful based on certain criteria.

The data for this analysis was compiled from more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME—Identification** columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

## Results
Deliverable 1: Preprocessing Data for a Neural Network Model
Using your knowledge of Pandas and the Scikit-Learn’s StandardScaler(), you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2.

The EIN and NAME columns have been dropped (5 pt)
The columns with more than 10 unique values have been grouped together (5 pt)
The categorical variables have been encoded using one-hot encoding (5 pt)
The preprocessed data is split into features and target arrays (5 pt)
The preprocessed data is split into training and testing datasets (5 pt)
The numerical values have been standardized using the StandardScaler() module (5 pt)


Deliverable 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Deliverable 3: Optimize the Model



## Summary