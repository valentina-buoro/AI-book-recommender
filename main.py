import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# import gradio as gr

from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

load_dotenv()

templates = Jinja2Templates(directory="templates")

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"]
)

raw_documents = TextLoader("tagged_descriptions.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db_books = Chroma.from_documents(documents, embedding=huggingface_embeddings)


def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    print(f"Found {len(recs)} similar entries for query: '{query}'")
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)
    print(f"Book recommendations:{book_recs[:2]}")

    if category == "All":
        book_recs = book_recs.head(final_top_k)
        print(f"third Book recommendations:{book_recs[:2]}")
    elif category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(
            final_top_k
        )
        print(f"Second Book recommendations:{book_recs[:2]}")

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprise":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)

    return book_recs


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/book-recommender")
def recommend_books(
    query: str,
    category: Literal[
        "All", "Fiction", "Nonfiction", "Children's Fiction", "Children's Nonfiction"
    ] = "All",
    tone: Literal["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"] = "All",
):
    try:
        recommendations = retrieve_semantic_recommendations(query, category, tone)
        keys = ["thumbnail", "caption", "description", "rating", "year", "pages", "id"]

        results = []
        for _, row in recommendations.iterrows():
            isbn = row["isbn13"]
            description = row["description"]
            average_rating = row["average_rating"]
            published_year = row["published_year"]
            num_of_pages = row["num_pages"]

            truncated_desc_split = description.split()
            truncated_description = " ".join(truncated_desc_split[:15]) + "..."

            authors_split = row["authors"].split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
            else:
                authors_str = row["authors"]
            caption = f"{row['title']} by {authors_str}: {truncated_description}"

            results.append(
                (
                    row["large_thumbnail"],
                    caption,
                    description,
                    average_rating,
                    published_year,
                    num_of_pages,
                    isbn,
                )
            )

        return [dict(zip(keys, item)) for item in results]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


"""

categories = ["All"] + sorted(books['simple_categories'].unique())
tones = ["All"] + ["Happy", "Surprising","Angry","Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("#Book Recommender")

    with gr.Row():
        with gr.Column():

            user_query = gr.Textbox(label="Please enter a book description:",
                               placeholder="e.g., A story about forgiveness", scale=2)
            with gr.Row():
                category_dropdown = gr.Dropdown(choices=categories,label="Select a category",value="All")
                tone_dropdown = gr.Dropdown(choices=tones, label="Select a tone:",value="All")
            submit_button = gr.Button(value="Find recommendation", scale = 1)

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommend Books", columns=4, rows=4)

    submit_button.click(fn = recommend_books,
                        inputs=[user_query,category_dropdown,tone_dropdown],outputs=output)
"""

"""
local development only
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)"""
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
