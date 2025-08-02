import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium", app_title="Book Recommender")

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Book Recommender
    Hi there! 👋
    This is me testing out the marimo framework with a bookish dataset I found on Kaggle. The dataset contains book titles and descriptions, and I will be using it to build a simple book recommender system.
    """
    )
    return

@app.cell
def _():
    import marimo as mo
    import polars as pl
    import torch
    from sentence_transformers import SentenceTransformer, util
    return SentenceTransformer, mo, pl, torch


@app.cell
def _(pl):
    # Loading DataFrame
    data = pl.read_csv('data/book.csv') #Adjust path as needed
    titles = data["title"].to_list()
    descriptions = data["description"].to_list()
    data.head()
    return descriptions, titles


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Since this is a pretty clean dataset that I pulled from Kaggle, we can dive right in. Firstly, I will try to find similar books based on the book descriptions. To do this I will train a sentence transformer. 

    Training Started ... (Aprrox. 2 minutes)
    """
    )
    return


@app.cell
def _(SentenceTransformer, descriptions):
    print('Training Started...')
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dataset_embeddings = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
    print('Training Completed...')
    return dataset_embeddings, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Training Completed!

    Next I will create a function that takes in the description of a book and returns a list of books similart to it and feed it the description of my favourite book of ALL time **The Covenant of Water** by *Abraham Verghese*.

    **Description**: Spanning the years 1900 to 1977, The Covenant of Water is set in Kerala, on India’s Malabar Coast, and follows three generations of a family that suffers a peculiar affliction: in every generation, at least one person dies by drowning—and in Kerala, water is everywhere. At the turn of the century, a twelve-year-old girl from Kerala's Christian community, grieving the death of her father, is sent by boat to her wedding, where she will meet her forty-year-old husband for the first time. From this unforgettable new beginning, the young girl—and future matriarch, Big Ammachi—will witness unthinkable changes over the span of her extraordinary life, full of joy and triumph as well as hardship and loss, her faith and love the only constants.
    """
    )
    return


@app.cell
def _(dataset_embeddings, model, pl, titles, torch):
    def recommend_from_input_description(input_desc: str, top_n: int = 5):
        # Suppress the warning by being more explicit with encoding
        input_embedding = model.encode(input_desc, convert_to_tensor=True, show_progress_bar=False)
        scores = torch.nn.functional.cosine_similarity(input_embedding, dataset_embeddings, dim=1)
        top_indices = torch.topk(scores, k=top_n).indices.tolist()

        results = [(titles[i], float(scores[i])) for i in top_indices]
        # Fix the DataFrame creation with explicit orientation
        return pl.DataFrame(results, schema=["Recommended Book", "Similarity Score"])
    return (recommend_from_input_description,)


@app.cell
def _(recommend_from_input_description):
    sample_description = """Spanning the years 1900 to 1977, The Covenant of Water is set in Kerala, on India’s Malabar Coast, and follows three generations of a family that suffers a peculiar affliction: in every generation, at least one person dies by drowning—and in Kerala, water is everywhere. At the turn of the century, a twelve-year-old girl from Kerala's Christian community, grieving the death of her father, is sent by boat to her wedding, where she will meet her forty-year-old husband for the first time. From this unforgettable new beginning, the young girl—and future matriarch, Big Ammachi—will witness unthinkable changes over the span of her extraordinary life, full of joy and triumph as well as hardship and loss, her faith and love the only constants."""

    recommend_from_input_description(sample_description, top_n=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Try it yourself! Enter a book description below and see the recommendations based on it.
    """
    )
    return

@app.cell
def _(mo):
    user_description = mo.ui.text(placeholder="Enter description...", label="Description")
    user_description
    return (user_description,)


@app.cell
def _(recommend_from_input_description, user_description):
    # Assign the result to a variable that marimo can display
    recommendations = None

    if user_description.value and len(user_description.value.strip()) > 10:
        recommendations = recommend_from_input_description(user_description.value)

    recommendations
    return


if __name__ == "__main__":
    app.run()
