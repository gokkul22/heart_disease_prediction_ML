# heart_disease_prediction_ML

# INTRODUCTION
In this project, we will be building a heart disease prediction
website which gives result for the data entered.We use an
online dataset which contains some medical information of
patients which tells whether that person getting a heart attack
chance is less or more. Using the information explore the
dataset and classify the target variable using different Machine
Learning models and find out which algorithm suitable for this
dataset and find the most accurate model.We use that model
in our webpage to predict whether the given new data has a
heart disease or not.

# ABSTRACT
Heart disease is a broad term used for diseases and
conditions affecting the heart and circulatory system. They are
also referred to as cardiovascular diseases. It is a major cause
of disability all around the world. Since the heart is amongst the
most vital organs of the body, its diseases affect other organs and
part of the body as well. There are several different types and
forms of heart diseases. The most common ones cause narrowing
or blockage of the coronary arteries, malfunctioning in the valves
of the heart, enlargement in the size of heart and several others
leading to heart failure and heart attack.Key facts according to
WHO (World Health Organizations) :
• Cardiovascular diseases (CVDs) are the leading cause of
death globally.
• An estimated 17.9 million people died from CVDs in 2019,
representing 32% of all global deaths. Of these deaths, 85%
were due to heart attack and stroke.
• Over three quarters of CVD deaths take place in low- and
middle-income countries.
• Out of the 17 million premature deaths (under the age of
70) due to noncommunicable diseases in 2019, 38% were
caused by CVDs.
• Most cardiovascular diseases can be prevented by addressing
behavioral risk factors such as tobacco use, unhealthy
diet and obesity, physical inactivity and harmful use of
alcohol.
• It is important to detect cardiovascular disease as early as
possible so that management with counseling and medicines
can begin.
Cardiovascular disease or heart disease describes a range of
conditions that affect your heart. Diseases under the heart
disease umbrella include blood vessel diseases, such as coronary
artery disease. From WHO statistics every year 17.9 million
die from heart disease. The medical study says that human
lifestyle is the main reason behind this heart problem. Apart
from this there are many key factors which warns that the person
may/may not be getting chance of heart disease.Using a dataset
containing information on patients who had health checkups, if
we create suitable machine learning techniques which predicts
the heart disease more accurately, it is very helpful to the health
organization as well as patients.

# PROPOSED WORK

A. Import Packages
• Pandas is a software library written for the Python programming
language for data manipulation and analysis.
In particular, it offers data structures and operations for
manipulating numerical tables and time series.
• NumPy is a library for the Python programming language,
adding support for large, multi-dimensional arrays
and matrices, along with a large collection of high-level
mathematical functions to operate on these arrays.
• Matplotlib is a comprehensive library for creating static,
animated, and interactive visualizations in Python.
• Scikit-learn (formerly scikits.learn and also known as
sklearn) is a free software machine learning library for
the Python programming language. It features various
classification, regression and clustering algorithms including
support-vector machines, random forests, gradient
boosting, k-means and DBSCAN, and is designed
to interoperate with the Python numerical and scientific
libraries NumPy and SciPy.
• The plotly Python library is an interactive, open-source
plotting library that supports over 40 unique chart types
covering a wide range of statistical, financial, geographic,
scientific, and 3-dimensional use-cases.
• Python pickle module is used for serializing and deserializing
python object structures. The process to converts
any kind of python objects (list, dict, etc.) into byte
streams (0s and 1s) is called pickling or serialization or
flattening or marshalling.
• These are the packages imported to use in our modeling
B. Data Preparation and Exploration
The dataset has 14 attributes:
• age: age in years.
• sex: sex (1 = male; 0 = female).
• cp: chest pain type (Value 0: typical angina; Value 1:
atypical angina; Value 2: non-anginal pain; Value 3:
asymptomatic).
• trestbps: resting blood pressure in mm Hg on admission
to the hospital.
• chol: serum cholestoral in mg/dl.
• fbs: fasting blood sugar ¿ 120 mg/dl (1 = true; 0 = false).
• restecg: resting electrocardiographic results (Value 0:
normal; Value 1: having ST-T wave abnormality; Value
2: probable or definite left ventricular hypertrophy).
• thalach: maximum heart rate achieved.
• exang: exercise induced angina (1 = yes; 0 = no)
• oldpeak: ST depression induced by exercise relative to
rest.
• slope: the slope of the peak exercise ST segment (Value
0: upsloping; Value 1: flat; Value 2: downsloping).
• ca: number of major vessels (0-3) colored by flourosopy.
• thal: thalassemia (3 = normal; 6 = fixed defect; 7 =
reversable defect).
• target: heart disease (1 = no, 2 = yes)
C. Preparing ML models
1) K-Nearest Neighbor:
• K-NN algorithm assumes the similarity between the new
case/data and available cases and put the new case into the
category that is most similar to the available categories.
• K-NN algorithm stores all the available data and classifies
a new data point based on the similarity. This means when
new data appears then it can be easily classified into a
well suite category by using K- NN algorithm.
• K-NN algorithm can be used for Regression as well as for
Classification but mostly it is used for the Classification
problems.
2) Random Forest Classifier:
• Random Forest is a classifier that contains a number of
decision trees on various subsets of the given dataset and
takes the average to improve the predictive accuracy of
that dataset.
• Random Forest can be used for both Classification and
Regression problems in ML.
• Random Forest is based on the concept of ensemble
learning, which is a process of combining multiple classifiers
to solve a complex problem and to improve the
performance of the model.
3) Support Vector Machine:
• The goal of the SVM algorithm is to create the best line or
decision boundary that can segregate n-dimensional space
into classes so that we can easily put the new data point
in the correct category in the future.This best decision
boundary is called a hyperplane.
• SVM chooses the extreme points/vectors that help in
creating the hyperplane.
• Support Vector machine is used for Classification as well
as Regression problems.
• SVM algorithm can be used for Face detection, image
classification, text categorization.
4) Decision Tree:
• In a Decision tree, there are two nodes, which are
the Decision Node and Leaf Node. Decision nodes are
used to make any decision and have multiple branches,
whereas Leaf nodes are the output of those decisions and
do not contain any further branches.
• It is a graphical representation for getting all the possible
solutions to a problem/decision based on given conditions.
• In a decision tree, for predicting the class of the given
dataset, the algorithm starts from the root node of the
tree. This algorithm compares the values of root attribute
with the record (real dataset) attribute and, based on the
comparison, follows the branch and jumps to the next
node.
• For the next node, the algorithm again compares the
attribute value with the other sub-nodes and move further.
It continues the process until it reaches the leaf node of
the tree.
5) Logistic Regression:
• Logistic regression predicts the output of a categorical
dependent variable. Therefore the outcome must be a
categorical or discrete value. It can be either Yes or No,
0 or 1, true or False, etc. but instead of giving the exact
value as 0 and 1, it gives the probabilistic values which
lie between 0 and 1.
• Logistic regression is used for solving the classification
problems.
• Logistic Regression is a significant machine learning
algorithm because it has the ability to provide probabilities
and classify new data using continuous and discrete
datasets.
6) Naive Bayes Classifier:
• Na¨ıve Bayes algorithm is a supervised learning algorithm,
which is based on Bayes theorem and used for solving
classification problems.
• Na¨ıve Bayes Classifier is one of the simple and most
effective Classification algorithms which helps in building
the fast machine learning models that can make quick
predictions.
• It is a probabilistic classifier, which means it predicts on
the basis of the probability of an object.
7) Extreme Gradient Boost:
• Gradient boosting refers to a class of ensemble machine
learning algorithms that can be used for classification or
regression predictive modeling problems.
• Extreme Gradient Boosting, or XGBoost for short is
an efficient open-source implementation of the gradient
boosting algorithm. As such, XGBoost is an algorithm,
an open-source project, and a Python library
D. Models evaluation and selection
• Evaluating accuracy of each model and then the best
model is found.
• Save the model as serialized object pickle.
E. WebPage Development
• A Webpage containing a form is created, the form contains
details to be entered by the user.
• The webpage uses the best model which we obtained after
comparing all the models for evaluating the user entered
data.
F. Prediction
• The user data is used to predict whether there is a heart
disease for the given user detail.
• It uses the best model which gives the most accuracy to
evaluate for the given data.
• It then predicts the result

