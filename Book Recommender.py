import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import torch
    from sentence_transformers import SentenceTransformer, util
    return SentenceTransformer, mo, pl, torch, util


@app.cell
def _(pl):
    # Loading DataFrame
    data = pl.read_csv('data/book.csv')
    titles = data["title"].to_list()
    descriptions = data["description"].to_list()
    data.head()
    return descriptions, titles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Since this is a pretty clean dataset that I pulled from Kaggle, I can dive right into the juicy stuff. Firstly, I will try to find similar books based on the book descriptions. For this purpose I will use word embeddings to extract the semantics of the text and then use cosine similarity to find the closeness of each book.""")
    return


@app.cell
def _(SentenceTransformer, descriptions):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dataset_embeddings = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
    return dataset_embeddings, model


@app.cell
def _(dataset_embeddings, model, pl, titles, torch, util):
    def recommend_from_input_description(input_desc: str, top_n: int = 5):
        input_embedding = model.encode(input_desc, convert_to_tensor=True)
        scores = util.cos_sim(input_embedding, dataset_embeddings)[0]
        top_indices = torch.topk(scores, k=top_n).indices.tolist()

        results = [(titles[i], float(scores[i])) for i in top_indices]
        return pl.DataFrame(results, schema=["Recommended Book", "Similarity Score"])

    return (recommend_from_input_description,)


@app.cell
def _(recommend_from_input_description):
    sample_description = """Spanning the years 1900 to 1977, The Covenant of Water is set in Kerala, on India’s Malabar Coast, and follows three generations of a family that suffers a peculiar affliction: in every generation, at least one person dies by drowning—and in Kerala, water is everywhere. At the turn of the century, a twelve-year-old girl from Kerala's Christian community, grieving the death of her father, is sent by boat to her wedding, where she will meet her forty-year-old husband for the first time. From this unforgettable new beginning, the young girl—and future matriarch, Big Ammachi—will witness unthinkable changes over the span of her extraordinary life, full of joy and triumph as well as hardship and loss, her faith and love the only constants."""

    recommend_from_input_description(sample_description, top_n=5)
    return


if __name__ == "__main__":
    app.run()
