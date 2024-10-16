# Streamlit_Apps

### 1. ANN for Customer Churn:
The dataset is loaded, categorical variables are encoded and the numerical variables are scaled which are then saved in pickle files. A simple neural network with 5 layers predicts the probability of churn with a 87% validation accuracy. Finally the application is deployed on Streamlit which predicts the outcome in real-time using the trained neural network.

### 2. RNN for Sentiment Analysis:
The IMDB movies dataset is loaded with a vocabulary of 50000 words and a simple RNN with max padding of 500 is trained on the dataset with 5 layers using the sigmoid activation function, with EarlyStopping monitoring the validation losses. Further, the neural network is deployed using Streamlit which predicts the sentiment score for the entered customer review.

### 3. Language Translation:
The language translator, deployed through Streamlit, is using Gemma2-9b model via Chat Groq. The Chat Prompt template takes in user text and translates it into the desired language.

### 4. PDF Retrieval-Augmented Generation:
The streamlit application loads in desired PDFs, creates vector embeddings via Gemma2-9b model and stores the vectors in FAISS database. Results are retrieved based on cosine similarity from the database.

### 5. Coder LLAMA:
The application uses Code LLAMA at backend to answer user coding questions. It loads a modelfile downloaded locally to call the API using gradio and displays the required code in various languages.

### 6. Text Summarization:
The text summarizer takes input url from the user, including Youtube and any website. It decodes the information present and outputs a concise summary of the website or Youtube video.
