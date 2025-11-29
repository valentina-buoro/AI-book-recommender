# AI-book-recommender

## Desription
This is a Semantic Book Recommender API. Unlike a standard search that looks for matching keywords, this application uses Vector Embeddings to understand the meaning or vibe of a user's query. It uses FastAPI for the backend/frontend, LangChain and ChromaDB for vector storage.

## How to run on your local machine
- clone the repository `git clone https://github.com/valentina-buoro/AI-book-recommender`
- Go into the folder `cd AI-book-recommender`
- create a virtual environment and activate it `python3 -m venv myenv (on macOS/linux) or python -m venv myenv (on windows)`.
- activate the virtual environment `source myenv/bin/activate (on macOS/linux) or myenv\Scripts\activate (on windows)`
- install project dependencies from `requirements.txt` `pip install -r requirements.txt`
- run the app `uvicorn main:app`
