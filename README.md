## Overview
This project implements an interactive financial QA bot that extracts data from P&L statements in PDFs, processes them using a Retrieval-Augmented Generation (RAG) pipeline, and provides AI-generated responses to user queries. The system uses Cohere for generative responses and Pinecone for vector-based retrieval.

## Features
- **Upload PDF files** containing financial statements.
- **Extract and preprocess** P&L data from PDFs.
- **Embed and store** financial terms for fast retrieval.
- **Ask financial queries** and receive AI-generated answers.
- **Deployable via Docker** for seamless scalability.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Docker (for containerization)

### Setup
Clone the repository and navigate to the project directory:
```sh
git clone https://github.com/your-repo/financial-qa-bot.git
cd financial-qa-bot
```

Install the required dependencies:
```sh
pip install -r requirements.txt
```

## Running the Application

### 1. Running Locally
To launch the Streamlit interface, run:
```sh
streamlit run app.py
```

### 2. Running with Docker
Build and run the container:
```sh
docker build -t financial-qa-bot .
docker run -p 8501:8501 financial-qa-bot
```

## Usage Guide
1. Open the application in your browser (default: `http://localhost:8501`).
2. Upload a PDF containing P&L tables.
3. Type a financial query, such as:
   - "What is the total revenue for 2023?"
   - "How do operating expenses compare for Q1 2024?"
4. View retrieved data and AI-generated responses.

## Example Queries
| Query | Response (Example) |
|--------|----------------|
| "What is the gross profit for Q3 2024?" | "The gross profit for Q3 2024 is $10M based on the extracted financial data." |
| "Show the operating margin for the past 6 months." | "The operating margin for the past 6 months is 25%." |

## Deployment
To deploy this app on a cloud service (AWS, GCP, etc.), ensure the following:
- The `Dockerfile` is set up for production.
- The API keys for Pinecone and Cohere are configured securely.
- The server environment supports Streamlit or Gradio.

## Contributing
Feel free to fork the repository and submit a pull request with improvements.
