# Swahili Horticulture Question Answering

A domain-specific question-answering assistant for **Swahili horticulture** (fruits, spices, vegetables). It retrieves a passage from farming manuals, extracts the answer with a neural reader, and **declines** questions it cannot answer.

The models taught in class (**BiGRU and BiLSTM with attention, and a Transformer encoder**) are built and trained from scratch, and compared with a pretrained multilingual Transformer. The best from-scratch model is saved in `app/model.pkl` and served by a **Streamlit** app that needs only `streamlit` and `numpy`.

## Project links
- GitHub repository: TODO
- Demo video (10 to 15 minutes): TODO
- Live app (Streamlit Community Cloud): TODO

## 1. Problem
Smallholder horticultural producers need practical farming information, and Swahili has few question-answering resources. The project asks: how well can retrieval plus a neural reader answer Swahili horticulture questions from farming manuals, how do models trained from scratch compare with pretrained ones on such a small dataset, and can the system tell when a question is outside the domain?

## 2. Data
| Role | Dataset | Licence |
|---|---|---|
| Train, validation, test | *Swahili Question-Answering Dataset for Horticulture* (Lubawa, 2024), Harvard Dataverse, DOI 10.7910/DVN/SORRLR | CC0 1.0 |
| Out-of-domain questions | AfriQA Swahili queries (Ogundepo et al., 2023) | CC BY-NC 4.0 (non-commercial use) |

2231 raw question and answer pairs over 307 passages, 2225 after cleaning. Passages are split into train, validation and test **by passage group** (overlapping passages are kept together) so no passage leaks between splits.

| Split | Questions | Passages |
|---|---|---|
| train | 1560 | 219 |
| val | 330 | 45 |
| test | 335 | 43 |

Out-of-domain questions after cleaning: 412 (validation) and 301 (test). Details: [docs/DATA_CARD.md](docs/DATA_CARD.md) and `results/data/`.

## 3. Method
1. **Retrieval:** BM25, written from scratch.
2. **Reading:** a model scores every passage word as the start and as the end of the answer; the best span is the answer.
3. **Refusal:** a confidence score (passage match and the reader's confidence) is compared with a threshold chosen on validation data.

**Systems compared** (all scored on the same validation and test questions, with the correct passage given ("oracle") and end-to-end through BM25):
- Baselines: BM25 + a rule-based span picker, and a pretrained multilingual QA model used zero-shot (E1 reads several passages; E3 trains only its last layer; E4 fine-tunes all layers on a GPU).
- **From scratch (E5):** BiGRU + attention, BiLSTM + attention, Transformer encoder. Word-level tokens, no pretrained files, 15 epochs, AdamW, early stopping on validation.
- **Deployed (E6):** the best from-scratch model by validation F1, with a refusal rule fitted on validation.

The full pipeline is in one notebook: [notebooks/swahili_horticulture_qa.ipynb](notebooks/swahili_horticulture_qa.ipynb). Every number in this README is read from files in `results/`.

## 4. Results
**Retrieval (BM25).** recall@1 on test: exact words 0.463, first 4 letters 0.427, first 5 letters 0.445, first 6 letters 0.460. Recall@10 (first 6 letters): validation 0.946, test 0.892. Prefix stemming did not clearly help.

**All systems** (EM / F1; oracle = correct passage given; BM25 top-1 = end-to-end):

| System | Setting | Val EM / F1 | Test EM / F1 |
|---|---|---|---|
| Rule-based reader | oracle passage | 0.000 / 0.105 | 0.003 / 0.093 |
| Rule-based reader | BM25 top-1 | 0.000 / 0.071 | 0.003 / 0.068 |
| Zero-shot XLM-R-base-SQuAD2 | oracle passage | 0.309 / 0.446 | 0.343 / 0.492 |
| Zero-shot XLM-R-base-SQuAD2 | BM25 top-1 | 0.167 / 0.272 | 0.170 / 0.263 |
| Zero-shot reader, top-k passages | BM25 top-3, choose by span score | 0.221 / 0.321 | 0.221 / 0.326 |
| Zero-shot reader, top-k passages | BM25 top-3, choose by margin | 0.224 / 0.326 | 0.218 / 0.326 |
| Zero-shot reader, top-k passages | BM25 top-5, choose by span score | 0.233 / 0.342 | 0.221 / 0.338 |
| Zero-shot reader, top-k passages | BM25 top-5, choose by margin | 0.233 / 0.344 | 0.215 / 0.330 |
| Head-only fine-tuned (frozen encoder) | oracle passage | n/a / 0.498 | 0.399 / 0.533 |
| BiGRU + attention (trained from scratch) | oracle passage | 0.140 / 0.237 | 0.133 / 0.236 |
| BiLSTM + attention (trained from scratch) | oracle passage | 0.144 / 0.227 | 0.123 / 0.222 |
| Transformer encoder (2 layers, 4 heads) (trained from scratch) | oracle passage | 0.031 / 0.100 | 0.016 / 0.090 |
| DEPLOYED: BiGRU + attention (seed 43) | oracle passage | 0.179 / 0.294 | 0.179 / 0.299 |
| DEPLOYED: BiGRU + attention (seed 43) | BM25 top-1 | 0.088 / 0.188 | 0.078 / 0.171 |
| Full fine-tune afroxlmr_lr3e-05_s42 | oracle passage | 0.327 / 0.438 | 0.358 / 0.483 |
| Full fine-tune afroxlmr_lr3e-05_s42 | BM25 top-1 | 0.191 / 0.277 | 0.185 / 0.264 |
| Full fine-tune xlmr_base_lr3e-05_s42 | oracle passage | 0.303 / 0.402 | 0.328 / 0.452 |
| Full fine-tune xlmr_base_lr3e-05_s42 | BM25 top-1 | 0.182 / 0.250 | 0.167 / 0.259 |
| Full fine-tune xlmr_squad2_lr1e-05_s42 | oracle passage | 0.470 / 0.604 | 0.492 / 0.634 |
| Full fine-tune xlmr_squad2_lr1e-05_s42 | BM25 top-1 | 0.258 / 0.366 | 0.236 / 0.338 |
| Full fine-tune xlmr_squad2_lr2e-05_s42 | oracle passage | 0.473 / 0.619 | 0.540 / 0.672 |
| Full fine-tune xlmr_squad2_lr2e-05_s42 | BM25 top-1 | 0.261 / 0.380 | 0.260 / 0.356 |
| Full fine-tune xlmr_squad2_lr2e-05_s43 | oracle passage | 0.512 / 0.647 | 0.534 / 0.673 |
| Full fine-tune xlmr_squad2_lr2e-05_s43 | BM25 top-1 | 0.279 / 0.382 | 0.254 / 0.355 |
| Full fine-tune xlmr_squad2_lr2e-05_s44 | oracle passage | 0.500 / 0.632 | 0.534 / 0.661 |
| Full fine-tune xlmr_squad2_lr2e-05_s44 | BM25 top-1 | 0.261 / 0.373 | 0.254 / 0.350 |
| Full fine-tune xlmr_squad2_lr4e-05_s42 | oracle passage | 0.506 / 0.638 | 0.558 / 0.676 |
| Full fine-tune xlmr_squad2_lr4e-05_s42 | BM25 top-1 | 0.288 / 0.398 | 0.269 / 0.362 |

**From-scratch models** (oracle passage, mean over seeds):

| Model | Parameters | Val F1 | Test EM | Test F1 (± sd) |
|---|---|---|---|---|
| BiGRU + attention | 1,501,698 | 0.237 | 0.133 | 0.236 ± 0.056 |
| BiLSTM + attention | 1,797,634 | 0.227 | 0.123 | 0.222 ± 0.068 |
| Transformer encoder (2 layers, 4 heads) | 929,794 | 0.100 | 0.016 | 0.090 ± 0.009 |

**Deployed model:** BiGRU + attention (seed 43, best epoch 14). Test set, 95% bootstrap intervals:

| Measure | Estimate |
|---|---|
| Oracle passage EM | 0.179 (0.137 to 0.221) |
| Oracle passage F1 | 0.299 (0.258 to 0.342) |
| End-to-end EM | 0.078 (0.051 to 0.107) |
| End-to-end F1 | 0.171 (0.139 to 0.205) |
| Retrieval recall@1 | 0.460 (0.409 to 0.513) |
| Refusal AUROC (combined (logistic regression)) | 0.964 (0.949 to 0.975) |
| In-domain questions answered | 90.5% |
| Out-of-domain questions refused | 89.0% |

Reference: the rule-based baseline reaches test F1 0.093 with the correct passage; the pretrained model used zero-shot reaches 0.492 (end-to-end 0.263). On test with the correct passage the deployed from-scratch model scores F1 0.299, which is below the zero-shot pretrained model (0.492). The deployed model was chosen for size and independence from pretrained files, not for being the most accurate. Error analysis: [results/error_analysis.md](results/error_analysis.md).

## 5. Repository layout
```
notebooks/   swahili_horticulture_qa.ipynb: the whole pipeline (data, baselines, experiments, evaluation, export)
app/         Streamlit app: streamlit_app.py, qa_pipeline.py (NumPy), model.pkl, passages.jsonl, requirements.txt
results/     every metric (JSON/CSV), figure, prediction file and the error analysis
docs/        decisions log, concepts guide, feasibility notes, viva questions, video script, guides
data/        downloaded by the notebook (not in the repository)
```

## 6. How to run
**Notebook (Google Colab with a T4 GPU, recommended):** upload the notebook to Google Drive, open it with Colab, set the runtime to T4 GPU, and run all cells. It mounts Drive and writes `results/` and `app/model.pkl` next to the notebook. Steps and time estimates: [docs/COLAB_RUN.md](docs/COLAB_RUN.md). Running on a CPU is possible but slow for the pretrained fine-tuning (E4) and for E5.

**Web app locally:**
```
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```
**Public link:** deploy `app/streamlit_app.py` on Streamlit Community Cloud from a public GitHub repository ([app/README.md](app/README.md), [docs/DEPLOY_AND_SUBMIT.md](docs/DEPLOY_AND_SUBMIT.md)).

**Reproducibility:** one random seed (42; 42, 43, 44 for repeated runs) is set for every random choice; choices use validation data only; the test set is used for reporting.

## 7. Limitations
Small data and test set (see the confidence intervals); one narrow domain; in-domain questions were written from the passages; the out-of-domain probe is general trivia (easy); the from-scratch models use word tokens (unknown words, no morphology); hand-chosen word lists not yet reviewed by a Swahili speaker. More in the report.

## 8. Acknowledgements, licences and AI assistance
Code: MIT (see LICENSE). Datasets keep their own licences (table above). The pretrained model `deepset/xlm-roberta-base-squad2` is CC BY 4.0; XLM-R and AfroXLMR are MIT. Libraries: PyTorch, NumPy, pandas, scikit-learn, matplotlib, Streamlit, Hugging Face Transformers (pretrained experiments only). Papers are listed in the report's reference section.

AI assistance: this project was developed with the help of an AI coding assistant (Claude Code) for code, drafting and checking. The author is responsible for understanding, running and verifying the work. [CONFIRM: wording to match course policy.]
