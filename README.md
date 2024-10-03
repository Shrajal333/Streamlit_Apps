# Streamlit_Apps
This repo contains a simpleton implementation of ANN and RNN to predict customer churn and movie review sentiment respectively.

### ANN for Customer Churn:
The dataset is loaded, categorical variables are encoded and the numerical variables are scaled which are then saved in pickle files. A simple neural network with 5 layers predicts the probability of churn with a 87% validation accuracy. Finally the application is deployed on Streamlit which predicts the outcome in real-time using the trained neural network.

![image](https://github.com/user-attachments/assets/9a3fd400-6969-4645-99e1-28577c8f8cd3)

### RNN for Sentiment Analysis:
The IMDB movies dataset is loaded with a vocabulary of 50000 words and a simple RNN with max padding of 500 is trained on the dataset with 3 layers using the sigmoid activation function, with EarlyStopping monitoring the validation losses. Further, the neural network is deployed using Streamlit which predicts the sentiment score for the entered customer review.

![image](https://github.com/user-attachments/assets/fb987df2-a66a-4983-9117-b77beb66b5fb)
