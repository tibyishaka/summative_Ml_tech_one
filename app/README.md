# Swahili Horticulture Question Answering (web app)

Ask a question in Swahili about growing fruit, spices or vegetables. The app retrieves a passage from farming-manual text, highlights the answer, and declines questions outside horticulture.

**Files**
- `streamlit_app.py`: the Streamlit interface (the file you point the host at)
- `qa_pipeline.py`: BM25 retrieval, the neural reader re-implemented in NumPy, and the refusal rule
- `model.pkl`: the deployed model (weights stored as plain arrays, vocabulary, settings). Written by the project notebook, Part 5
- `passages.jsonl`: the 307 passages the app searches
- `requirements.txt`: `streamlit` and `numpy`. Nothing else: no PyTorch, no pretrained models

**Model:** the best of three networks trained from scratch on the project data (BiGRU with attention, BiLSTM with attention, Transformer encoder), chosen by validation F1. The notebook records which one won.
**Data:** Lubawa, A. (2024), Swahili Question-Answering Dataset for Horticulture, Harvard Dataverse, https://doi.org/10.7910/DVN/SORRLR (CC0 1.0).

## Run locally
```
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Get a public link (Streamlit Community Cloud, free)
Streamlit Community Cloud deploys an app straight from a GitHub repository and gives it a public URL. You need a GitHub account and a Streamlit Community Cloud account (sign in with GitHub).
1. Put the whole project on GitHub as a **public** repository (see `docs/DEPLOY_AND_SUBMIT.md`). Make sure `app/model.pkl` and `app/passages.jsonl` are in it; both are small.
2. Go to https://share.streamlit.io, click **Create app**, choose the repository and branch.
3. Set **Main file path** to `app/streamlit_app.py`.
4. Click **Deploy**. The first build takes a few minutes. The app URL is your public link.

Dependencies: Community Cloud looks for the requirements file in the folder of the main file first and then in the repository root, so `app/requirements.txt` is found. Source: [Streamlit docs, app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) and [file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

Free apps go to sleep when nobody uses them; open the link just before a demo or the deadline to wake it.

## Safety note about `model.pkl`
Pickle files can run code when loaded if they were tampered with. `qa_pipeline.load_bundle` refuses every class or function lookup, so a file that is not plain data is rejected. Still, only use a `model.pkl` that you produced yourself.

Not agronomic advice. The app only knows its 307 passages and can be wrong.
