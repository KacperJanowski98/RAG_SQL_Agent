--

# Intelligent SQL-agent and RAG System for Chatting with Multiple Databases

This project demonstrates how to build an agentic system using Large Language Models (LLMs) that can interact with multiple databases and utilize various tools. It highlights the use of SQL agents to efficiently query large databases. The key frameworks used in this project include OpenAI, LangChain, LangGraph, LangSmith, and Gradio. The end product is an end-to-end chatbot, designed to perform these tasks, with LangSmith used to monitor the performance of the agents.

---

### Inspiration
This project was made based on a YouTube video:

Automating LLM Agents to Chat with Multiple/Large Databases (Combining RAG and SQL Agents): [Link](https://youtu.be/xsCedrNP9w8?si=v-3k-BoDky_1IRsg)

---

### Requirements

- **Tavily Credentials:** Required for search tools (Free from your Tavily profile).
- **LangChain Credentials:** Required for LangSmith (Free from your LangChain profile).
- **Dependencies:** The necessary libraries are provided in `requirements.txt` file.
---

#### TODO

Added description of creating a vector database and added link to travel.sqlite download source.
Note: I used a chinook.db because there was an error with travel.sqlite database.
Additionally for testing I downloaded another sqlite database and there is no problem with functioning: https://www.kaggle.com/code/dimarudov/data-analysis-using-sql/notebook

#### Notes

I am using local LLM qwen2.5:14b. Its performance is really good but it has problems with SQL agent (problem with creating SQL queries).

Information about the database used: https://www.sqlitetutorial.net/sqlite-sample-database/
