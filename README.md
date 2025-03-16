# Medical Fraud Detection Application
## Overview
This repository contains the implementation of Medical invoices Fraud Detection App using ML . Application is made using Flask , Ml model used are Logistric Regression, Random Forest and XGBoost . The Application is connected with Postgres Database to store the results and the application is deployed on AWS EC2 service of which I  provided the application link in about.
## Repository Structure
- **App**: Contains the flask app code with the frontend html code.
- **Multiple ipynb files** : There is 4 ipynb files for ML training,datapreprocessing ,model testing and one more for converting the csv data to pdf .
- **pdfs** : Contains pregenerated pdfs from the trusted dataset to test the model/app.
- **Medical Dataset** : This folder contained dataset but its not there I'll provide the link to download dataset in this folder . The dataset is collected from kaggle link: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data

## Models

### 1. **Deep Q-Network (DQN)**

The DQN model is a reinforcement learning algorithm where a neural network is used to approximate the Q-value function. This repository provides an implementation of the DQN algorithm along with visualizations of the training process.

#### DQN Loss and Score

- **Loss**: The DQN loss graph shows the fluctuations in loss over episodes. Despite the high variance in loss initially, the overall trend indicates that the loss decreases as the agent learns.
- **Score**: The score graph shows the cumulative reward obtained by the agent over episodes. The agent's performance improves significantly after the initial exploration phase.

![DQN Loss and Score](./DQN.png)

### 2. **Duelling DQN**

The Duelling DQN architecture improves upon the DQN by estimating the state value and advantage separately. This helps in stabilizing the training process and achieving better performance.

#### Duelling DQN Loss and Score

- **Loss**: The loss for Duelling DQN shows similar high variance but generally decreases over time, indicating learning and adaptation.
- **Score**: The performance (score) improves steadily, reaching higher scores faster compared to the standard DQN.

![Duelling DQN Loss and Score](./Duelling%20DQN.png)

### 3. **Duelling DDQN**

The Duelling DDQN combines the Duelling architecture with Double DQN, which mitigates the overestimation bias present in Q-learning.

#### Duelling DDQN Loss and Score

- **Loss**: The loss graph for Duelling DDQN exhibits fluctuations similar to the other models but with some stability after initial training.
- **Score**: The score graph shows improved performance with more stable and higher scores compared to the DQN and Duelling DQN.

![Duelling DDQN Loss and Score](./duelling%20ddqn.png)

## Results

The figures show that both the Duelling DQN and Duelling DDQN architectures generally perform better than the standard DQN in terms of stability and score. The Duelling DDQN in particular demonstrates robust performance with the highest and most stable scores over the episodes.

## How to Run

1. Clone the repository.
2. Ensure you have all the required dependencies installed.
3. Run the script using:
   ```bash
   python dqn.py
