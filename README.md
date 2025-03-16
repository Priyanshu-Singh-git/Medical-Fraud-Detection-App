# Medical Fraud Detection Application
## Overview
### This repository contains the implementation of Medical invoices Fraud Detection App using ML . Application is made using Flask , Ml model used are Logistric Regression, Random Forest and XGBoost . The Application is connected with Postgres Database to store the results and the application is deployed on AWS EC2 service of which I  provided the application link in about.
## Repository Structure
- **App**: Contains the flask app code with the frontend html code.
- **Multiple ipynb files** : There is 4 ipynb files for ML training,datapreprocessing ,model testing and one more for converting the csv data to pdf .
- **pdfs** : Contains pregenerated pdfs from the trusted dataset to test the model/app.
- **Medical Dataset** : This folder contained dataset but its not there I'll provide the link to download dataset in this folder . The dataset is collected from kaggle link: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data

## Explanation of the Full Development procedure

### 1. **Medical Data Preprocessing**
#### The collected unprocessed data had three dataset - inpatient, outpatient and beneficiary. 
#### The inpatient and outpatient is the Claims given by the hospital side to the government claiming the Medical services details inpatient for the admitted patients and outpatients for minor injuries patient who visited just fro consultation and normal checkups.
 #### The Beneficiary contains the patients details the Bill ammount paid by him , the insurance details ,his claimed service details, and his medical record and past disease records. All these dataset are processed in the Preprocessing ipynb file . 
#### The data are merged on basis of the Beneficiary-ID(representing each unique patients ) and the Provider_ID(representing the set of unique hospitals) . You can check dataset authentication and verification in the kaggle link.


### 2. *Model Selection and training*
#### The Model selection was done by testing with three different Models Logistic Regression , Random forest Regression , XGBoost model .
#### From which the XGboost gave a very great metrics for both classes for Recall,Precision and Accuracy. Hyperparameter tuning was done like each the branches (n_estimators in case of Random Forest and changing the converge algo for logistic reg to Newton-cg).
####  Serialized the models using Pickle module of python

### 3. **Model testing**
#### Loading the model to check for whether the results are authentic or not (to also check isnt the model overfitted /underfitted)

### 4. **Application Development and Database integration**
#### Made the flask Application where you can drag and drop the invoice and the text would be extracted then formatted according to model.
#### After clicking on Predict button the model will predict the class alongwith showing prediction probability for more insights
#### On backend it would be connected to PostGresSQL which will after doing prediction will store the results having following columns - (Beneficiary id , Provider id, Extracted details, Predictions)

### 5. **Model and Application deployment to AWS EC2**
#### The deployment process was the one taking much time , had to create instance of EC2 (having linux AMI) .
#### Installed all dependencies(python and its required modules)
#### Transferred files to virtual cloud system then launched it Using tools like gunicorn, etc.


## Results

Model results are 
### Random forest results

|          | precision | recall | f1-score | support   | 
|----------|----------|----------|----------|----------|
| 0        | 0.86     | 0.95     |  0.90    | 68983    |
| 1        | 0.91     | 0.75     | 0.82     | 42660    |
| accuracy | ----     | ----     | 0.88     | 111643   |
| macro avg| 0.88     | 0.85     | 0.86     | 111643   |
|weight.avg| 0.88     | 0.88     | 0.87     | 111643   |

### XGboost results

|          | precision | recall | f1-score | support   | 
|----------|----------|----------|----------|----------|
| 0        | 0.94     | 0.97     |  0.95    | 68983    |
| 1        | 0.95     | 0.90     | 0.92     | 42660    |
| accuracy | ----     | ----     | 0.94     | 111643   |
| macro avg| 0.94     | 0.93     | 0.94     | 111643   |
|weight.avg| 0.94     | 0.94     | 0.94     | 111643   |

## How to Run
### Clone the repository 
### Download the dataset from the mentioned link
### After cloning change the model weights variable in app.py if deploying on cloud
### Also change the Postgres connection variables like user,host, password and port etc
### Change the parameter and retrain the models according to your wish 

