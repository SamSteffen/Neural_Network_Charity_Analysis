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
### Preprocessing
To build our binary classifier, we first cleaned the data in preparation for processing. This involved:
- dropping unnecessary data columns;
- grouping together columns that had more than 10 unique values;
- encoding categorical variables to integers using the one-hot coding method;
- splitting the pre-processed data into features and target arrays;
- splitting the pre-processed data into training and testing datasets;
- standardizing the numerical datasets using StandardScaler().

The target variables for our dataset are:
- **"IS SUCCESSFUL"**

The features of our model include:
- **"STATUS"**
- **"ASK_AMT"**
- **"APPLICATION_TYPE"** 
- **"INCOME_AMT"**
- **"SPECIAL_CONSIDERATIONS"**
- **"AFFILIATION"**
- **"CLASSIFICATION"**
- **"USE_CASE"**
- **"ORGANIZATION"**

The variables that are neither targets nor features include:
- **"EIN"**
- **"NAME-Identification"**

A visual of our dataframe, following preprocessing, prior to compiling and training, is shown below:

[df_visual]

### Compiling, Training, and Evaluating the Model
Once our data was preprocessed, we utilized TensorFlow to design a neural network (deep learning model) to create a binary classification model to predict whether organizations funded by Alphabet Soup would be successful, based on the features in the dataset.

For the hidden layers, our deep learning model used 2 hidden layers with 80 neurons in the first and 30 neurons in the second. These numbers were selected in the hopes of achieving a high accuracy statistic. All of the hidden layers used the relu activation function to identify nonlinear characteristics from the input values.

To determine the inputs, we used the same number of variables present in our feature DataFrame, of which there were 43.

Once the inputs, neurons, and layers were decided upon, we compiled, trained and evaluated our binary classification model to calculate the model's loss accuracy. The results of this process are shown below.

To design our model's output layer we used the "sigmoid" activation function to help us predict the probability that an organization receiving donations would be successful. A summary of the structure of our model is shown below:

[summary pic]

Looking at our model summary, we can see that the number of weight parameters (weight coefficients) for each layer equals the number of input values times the number of neurons plus a bias term for each neuron. Our first layer has 43 input values, and multiplied by the 80 neurons (plus eighty bias terms for each neuron) gives us a total of 3520 weight parameters—plenty of opportunities for our model to find trends in the dataset.

Since we wanted to use our model as a binary classifier, we then compiled our model using the ```binary_crossentropy``` loss function, ```adam``` optimizer, and ```accuracy``` metrics. We then evaluated our model's performance by testing its predictive capabilities on our testing dataset. The output of this evaluation is shown below.

[image]

## Summary
Looking at our deep learning model's performance metrics, the model was able to correctly identify successful donor recipients approximately 72% of the time. Considering that our input data included more than 43 different variables with more than 34,000 data points, the deep learning model was able to produce a fairly reliable classifier.

Once we created this model, we tried to optimize it in order to achieve a target predictive accuracy of higher than 75%.

Things we tried to increase the accuracy of our model performance, 

Were you able to achieve the target model performance?

What steps did you take to try and increase model performance?

Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
