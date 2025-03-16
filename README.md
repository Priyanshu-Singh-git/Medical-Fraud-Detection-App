# Medical Fraud Detection Application
## Overview
This repository contains the implementation of Medical invoices Fraud Detection App using ML . Application is made using Flask , Ml model used are Logistric Regression, Random Forest and XGBoost . The Application is connected with Postgres Database to store the results and the application is deployed on AWS EC2 service of which I  provided the application link in about.
## Repository Structure
- **App**: Contains the flask app code with the frontend html code.
- **Multiple ipynb files** : There is 4 ipynb files for ML training,datapreprocessing ,model testing and one more for converting the csv data to pdf .
- **pdfs** : Contains pregenerated pdfs from the trusted dataset to test the model/app.
- **Medical Dataset** : This folder contained dataset but its not there I'll provide the link to download dataset in this folder . The dataset is collected from kaggle link: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data

## Explanation of the Full Development procedure

### 1. **Medical Data Preprocessing**
The collected unprocessed data had three dataset - inpatient, outpatient and beneficiary. The inpatient and outpatient is the Claims given by the hospital side to the government claiming the Medical services details inpatient for the admitted patients and outpatients for minor injuries patient who visited just fro consultation and normal checkups. The Beneficiary contains the patients details the Bill ammount paid by him , the insurance details ,his claimed service details, and his medical record and past disease records. All these dataset are processed in the Preprocessing ipynb file . The data are merged on basis of the Beneficiary-ID(representing each unique patients ) and the Provider_ID(representing the set of unique hospitals) . You can check dataset authentication and verification in the kaggle link.
### 2. **Duelling DQN**






### 3. **Duelling DDQN**







## Results

The figures show that both the Duelling DQN and Duelling DDQN architectures generally perform better than the standard DQN in terms of stability and score. The Duelling DDQN in particular demonstrates robust performance with the highest and most stable scores over the episodes.

## How to Run

1. Clone the repository.
2. Ensure you have all the required dependencies installed.
3. Run the script using:
   ```bash
   python dqn.py
