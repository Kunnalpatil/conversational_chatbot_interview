# Conversational Chatbot Interview

Try the app : [https://conversationalchatbotinterview.streamlit.app/](https://conversationalchatbotinterview.streamlit.app/)

This repository contains a conversational chatbot designed for interview purposes. It is implemented in Python using Streamlit and various language models.

## Introduction

The purpose of this project is to provide a chatbot that can be used to to answer questions on my behalf.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```


Functionality
- The chatbot utilizes language model to provide accurate and context-aware responses.
- It maintains a session history to provide context to the conversation.
- It supports text-to-speech functionality to convert the response text into speech.

Features
1. Session Management: Start a new session by entering a session ID.
2. Embeddings and Vector Database: Create embeddings for the documents provided as context and store them in a vector database.
3. Contextual Question Answering: Reformulate questions based on the conversation history.
4. Speech Synthesis: Convert text responses to speech and play them automatically.
