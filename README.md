# Diabetes Prediction System

This project is a Diabetes Prediction System developed using Python and popular libraries such as NumPy, Pandas, Scikit-learn, Seaborn, and Matplotlib. The system is implemented in a Jupyter Notebook file.

## Table of Contents
- Introduction
- Installation
- Usage
- Data
- Methods
- Results
- Contributing

## Introduction
The Diabetes Prediction System is designed to predict the presence of diabetes in patients based on various medical features. The system utilizes a machine learning approach to analyze a dataset containing information about patients' health and predict whether an individual is likely to have diabetes or not. The prediction is made using the classification algorithm provided by the Scikit-learn library.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/vinod-polinati/Diatebetes-Prediction
   ```
2. Navigate to the project directory:
   ```
   cd Diabetes-Prediction-system
   ```
3. Install the required dependencies:
   ```
   pip install numpy pandas scikit-learn seaborn matplotlib jupyter
   ```

## Usage
1. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```
2. Open the `Diatebetes-Prediction using Machine Learning.ipynb` notebook.
3. Execute the cells in the notebook to run the code.

## Data
The dataset used in this project is the [NIKHIL ANAND Diabetes Database](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) from Kaggle. 
The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative). The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information.

## Methods
1. Data Preprocessing: The dataset is preprocessed to handle missing values, normalize the features, and split the data into training and testing sets.
2. Model Training: A machine learning model is trained on the training data using the Scikit-learn library. In this project, a logistic regression model is used for classification.
3. Model Evaluation: The trained model is evaluated on the testing data using various evaluation metrics such as accuracy, precision, recall, and F1 score.
4. Prediction: The trained model is then used to predict the presence of diabetes in new, unseen data.

## Results
The project provides an evaluation of the trained model's performance on the testing data. The evaluation metrics include accuracy, precision, recall, and F1 score. These metrics provide insights into how well the model is performing in predicting diabetes. We can see the AUC Score of every model and on every situations like on without scaled imbalanced data, scaled but imbalanced data and balanced data. After doing checking every model we can see that the AUC score of models before scaling on imbalanced data is better than imablanced data after scaling. And at last we found that balancing our data every model result is far better than all previous results.


<img width="1134" alt="Screenshot 2023-07-07 at 3 42 22 AM" src="https://github.com/vinod-polinati/Diatebetes-Prediction/assets/108022173/1e9bc8c5-69ad-4f33-a9c6-33b29a9b8ae7">




## Contributing
Contributors are always welcomed 🫶
