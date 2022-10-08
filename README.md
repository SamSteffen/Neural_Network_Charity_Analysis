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

<img width="659" alt="del_1_df" src="https://user-images.githubusercontent.com/104729703/194716366-71b8b439-0f38-498b-a617-ddb588e3c33c.png">

### Compiling, Training, and Evaluating the Model
Once our data was preprocessed, we utilized TensorFlow to design a neural network (deep learning model) to create a binary classification model to predict whether organizations funded by Alphabet Soup would be successful, based on the features in the dataset.

* For the hidden layers, our deep learning model used 2 hidden layers with 80 neurons in the first and 30 neurons in the second. These numbers were selected in the hopes of achieving a high accuracy statistic. All of the hidden layers used the relu activation function to identify nonlinear characteristics from the input values.

* To determine the inputs, we used the same number of variables present in our feature DataFrame, of which there were 43.

* Once the inputs, neurons, and layers were decided upon, we compiled, trained and evaluated our binary classification model to calculate the model's loss accuracy. 

* To design our model's output layer we used the "sigmoid" activation function to help us predict the probability that an organization receiving donations would be successful. A summary of the structure of our model is shown below:

<img width="451" alt="del_2_nn_summary" src="https://user-images.githubusercontent.com/104729703/194716370-98574158-c436-4a5b-83c1-3782be9a5863.png">

Looking at our model summary, we can see that the number of weight parameters (weight coefficients) for each layer equals the number of input values times the number of neurons plus a bias term for each neuron. Our first layer has 43 input values, and multiplied by the 80 neurons (plus eighty bias terms for each neuron) gives us a total of 3520 weight parameters—plenty of opportunities for our model to find trends in the dataset.

Since we wanted to use our model as a binary classifier, we then compiled our model using the ```binary_crossentropy``` loss function, ```adam``` optimizer, and ```accuracy``` metrics. We then evaluated our model's performance by testing its predictive capabilities on our testing dataset. The output of this evaluation is shown below.

<img width="661" alt="Evaluation" src="https://user-images.githubusercontent.com/104729703/194716378-d45d983c-d81d-4441-aa03-803506622d37.png">

## Summary
Looking at our deep learning model's performance metrics, the model was able to correctly identify successful donor recipients approximately 72% of the time. Considering that our input data included more than 43 different variables with more than 34,000 data points, the deep learning model was able to produce a fairly reliable classifier.

Our model had an optimization target predictive accuracy of higher than 75%. To see if we could achieve this accuracy score, we attempted to modify our model using the following methods: 

- Because our initial investigation of the "ASK_AMT" feature showed that it contained 8,747 unique variables, it was removed from the dataset.
- Increased the number of neurons from 80 to 100 in our first layer and from 30 to 50 in our second.
- We also added an additional hidden layer with 20 neurons.
- Changed the activation functions in our first layer to "sigmoid", in our second layer to "tanh" and set the third hidden layer activation function to "relu." The output layer activation remained at the "sigmoid" setting. 

These efforts yielded the following results:

### OPTIMIZATION ATTEMPT #1

<img width="735" alt="Evaluation_opt1" src="https://user-images.githubusercontent.com/104729703/194716387-a734ee69-3c3e-4e19-90d1-e3cd577c580e.png">

The output shows that this model was able to correctly identify successful donor recipients approximately 73% of the time. While this yielded a fairly reliable classifier, it is still short of our optimization goal of 75%.

### OPTIMIZTION ATTEMPT #2
For a second attempt at optimization, we made the following modifications to our model:
- Made all the neurons the same for each hidden layer and reset the activation functions to "relu" for our input, "relu" for the first hidden layer, "relu" for the second hidden layer, "sigmoid" for our third hidden layer, and "relu" for the output layer activation function.

These settings yielded the following output:

<img width="662" alt="Evaluation_opt2" src="https://user-images.githubusercontent.com/104729703/194716413-57ac9065-b8c9-43ed-b303-857427b54118.png">

### OPTIMIZATION ATTEMPT #3
For our third attempt at optimization, we made the following modifications to our model:
- Dropped additional features from the dataset
- Kept the 3 hidden layers from our previous attempts, but changed the number of neurons in each to 80 for the first, 50 for the second and 30 for the third.
- Changed the activation functions for each layer to "relu" for the first, second and third layers, and changed the output layer to "sigmoid."

These settings yielded the following output:

<img width="667" alt="Evaluation_opt3" src="https://user-images.githubusercontent.com/104729703/194716420-07394316-3cd8-49e1-9e23-70d4008cf34a.png">

These attempts show that we were unsuccessful in our attempt to develop a classifier that achieved a 75% accuracy score for determining whether an organization who receieved funding from the non-profit Alphabet Soup, would be successful in their endeavors. We attempted to increase model the model's performance by reducing the number of features, changing the number of neurons, adding additional hidden layers, and changing the activation functions. It is likely that the model is not performing well becuase of the sheer amount and variety of data. For further testing, it would be my recommendation to try reducing the features further and utilize a supervised learning model on a portion of the dataset to determine what variables are likely to impact the metric for "success." Further definition around this metric would also be useful in clarifying further testing.
