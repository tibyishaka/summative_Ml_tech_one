"""Swahili horticulture question answering with NumPy only (no PyTorch, no Hugging Face).

The pipeline has three steps, mirroring the project notebook (Part 2 and Part 3):
  1. BM25 retrieval picks the most relevant passage from the collection.
  2. A word-level neural reader (BiGRU, BiLSTM or Transformer encoder) scores every passage word as the start and as the
     end of the answer; the best span is the answer. The network was trained in PyTorch, exported into model.pkl as plain
     arrays, and is re-implemented here with NumPy so the web app needs no deep-learning library.
  3. A confidence rule (BM25 score + the reader's confidence, with a threshold chosen on validation data) declines
     questions that the manuals probably cannot answer.

model.pkl holds only built-in Python types (dicts, lists, strings, numbers, bytes), never custom classes, so it loads anywhere.
Pickle can execute code when loading a tampered file, so load_bundle() below refuses every class or function lookup: a file that
needs one is rejected instead of run. Even so, only use a model.pkl that you created yourself.
"""
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).parent
WORD = re.compile(r"\w+(?:'\w+)*", re.UNICODE)       # a word: letters/digits plus internal apostrophes (ng'ombe)
MAX_QUESTION_CHARS = 300
PAD_ID, UNK_ID, SEP_ID = 0, 1, 2


class _PlainDataUnpickler(pickle.Unpickler):
    """Allows plain data only. A pickle that tries to import any class or function (the way pickle attacks work) is rejected."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"model.pkl may only contain plain data, but it asked for {module}.{name}")


def load_bundle(path):
    with open(path, "rb") as f:
        return _PlainDataUnpickler(f).load()


# ---------------------------------------------------------------- BM25 (same formula as the notebook)
class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b, self.n = k1, b, len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype=float)
        self.avgdl = self.doc_len.mean()
        self.tf = [{} for _ in docs_tokens]
        doc_freq = {}
        for i, d in enumerate(docs_tokens):
            for t in d:
                self.tf[i][t] = self.tf[i].get(t, 0) + 1
            for t in set(d):
                doc_freq[t] = doc_freq.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in doc_freq.items()}
        self.idf_unseen = math.log(1 + (self.n + 0.5) / 0.5)

    def scores(self, q_tokens):
        s = np.zeros(self.n)
        for t in set(q_tokens):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
        return s

    def normalized_top_score(self, q_tokens, raw_top):
        best_possible = sum(self.idf.get(t, self.idf_unseen) * (self.k1 + 1) for t in set(q_tokens))
        return raw_top / best_possible if best_possible else 0.0


# ---------------------------------------------------------------- small NumPy building blocks
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, w, b, eps=1e-5):
    mean, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def log_softmax(v):
    m = v.max()
    return v - (m + np.log(np.exp(v - m).sum()))


# ---------------------------------------------------------------- the reader network
class NumpyReader:
    """Runs the exported network on one question + passage. Weight names follow PyTorch's state_dict names."""

    def __init__(self, bundle):
        self.arch = bundle["arch"]
        self.w = {k: np.frombuffer(v["data"], dtype=v["dtype"]).reshape(v["shape"]).astype(np.float64) for k, v in bundle["weights"].items()}

    # one direction of a GRU or LSTM over a whole sequence (x: length x input size)
    def _direction(self, x, prefix, suffix, cell):
        w = self.w
        Wih, Whh = w[f"{prefix}.weight_ih_l0{suffix}"], w[f"{prefix}.weight_hh_l0{suffix}"]
        bih, bhh = w[f"{prefix}.bias_ih_l0{suffix}"], w[f"{prefix}.bias_hh_l0{suffix}"]
        H = Whh.shape[1]
        gi = x @ Wih.T + bih                                  # input part of every gate, for all positions at once
        h, c = np.zeros(H), np.zeros(H)
        out = np.zeros((x.shape[0], H))
        order = range(x.shape[0] - 1, -1, -1) if suffix else range(x.shape[0])
        for t in order:
            gh = Whh @ h + bhh
            if cell == "gru":                                 # gates: reset r, update z, candidate n
                r = sigmoid(gi[t, :H] + gh[:H])
                z = sigmoid(gi[t, H:2 * H] + gh[H:2 * H])
                n = np.tanh(gi[t, 2 * H:] + r * gh[2 * H:])
                h = (1 - z) * n + z * h
            else:                                             # LSTM gates: input i, forget f, candidate g, output o
                g = gi[t] + gh
                i_, f_, g_, o_ = sigmoid(g[:H]), sigmoid(g[H:2 * H]), np.tanh(g[2 * H:3 * H]), sigmoid(g[3 * H:])
                c = f_ * c + i_ * g_
                h = o_ * np.tanh(c)
            out[t] = h
        return out

    def _birnn(self, x, prefix, cell):
        return np.concatenate([self._direction(x, prefix, "", cell), self._direction(x, prefix, "_reverse", cell)], axis=1)

    def _rnn_reader(self, ids, seg):
        w, cell = self.w, self.arch["kind"]
        qm = seg == 0                                         # question tokens (including the separator) are what attention may look at
        x = w["emb.weight"][ids] + w["seg.weight"][seg]
        H = self._birnn(x, "enc", cell)
        scores = H @ H.T
        scores[:, ~qm] = -1e4
        c = softmax(scores) @ H                               # attention: weighted sum of question vectors for every token
        M = self._birnn(np.concatenate([H, c, H * c], axis=1), "model", cell)
        return M @ w["out.weight"].T + w["out.bias"]

    def _transformer_reader(self, ids, seg):
        w, a = self.w, self.arch
        L, d, heads = len(ids), a["d"], a["heads"]
        dh = d // heads
        x = w["emb.weight"][ids] + w["pos.weight"][np.arange(L)] + w["seg.weight"][seg]
        for l in range(a["layers"]):
            p = f"enc.layers.{l}."
            qkv = x @ w[p + "self_attn.in_proj_weight"].T + w[p + "self_attn.in_proj_bias"]
            q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(L, heads, dh).transpose(1, 0, 2) for i in range(3))
            att = softmax(q @ k.transpose(0, 2, 1) / math.sqrt(dh))          # every token attends to every token, per head
            o = (att @ v).transpose(1, 0, 2).reshape(L, d)
            x = layer_norm(x + o @ w[p + "self_attn.out_proj.weight"].T + w[p + "self_attn.out_proj.bias"], w[p + "norm1.weight"], w[p + "norm1.bias"])
            ff = np.maximum(x @ w[p + "linear1.weight"].T + w[p + "linear1.bias"], 0) @ w[p + "linear2.weight"].T + w[p + "linear2.bias"]
            x = layer_norm(x + ff, w[p + "norm2.weight"], w[p + "norm2.bias"])
        return x @ w["out.weight"].T + w["out.bias"]

    def logits(self, ids, seg):
        """ids, seg: lists of ints for question + separator + passage. Returns an array (length, 2): start and end scores."""
        ids, seg = np.asarray(ids), np.asarray(seg)
        return self._transformer_reader(ids, seg) if self.arch["kind"] == "transformer" else self._rnn_reader(ids, seg)


# ---------------------------------------------------------------- the whole question-answering pipeline
class SwahiliHorticultureQA:
    def __init__(self, bundle_path=None, passages_path=None):
        self.bundle = load_bundle(bundle_path or APP_DIR / "model.pkl")
        rows = [json.loads(line) for line in Path(passages_path or APP_DIR / "passages.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.pids = [r["pid"] for r in rows]
        self.contexts = {r["pid"]: r["context"] for r in rows}
        k = self.bundle["retrieval"]["stem_k"]
        self.stem = (lambda w: w.lower()[:k]) if k else (lambda w: w.lower())
        self.bm25 = BM25([self.tokens(r["context"]) for r in rows])
        self.reader = NumpyReader(self.bundle)
        self.vocab = self.bundle["vocab"]
        self.max_answer_words = self.bundle["max_answer_words"]
        self.examples = self.bundle.get("examples", [])
        self.info = self.bundle.get("training", {})

    # ---- retrieval
    def tokens(self, text):
        return [self.stem(w) for w in WORD.findall(text)]

    def retrieve(self, question):
        q = self.tokens(question)
        s = self.bm25.scores(q)
        i = int(np.argsort(-s, kind="stable")[0])
        return self.pids[i], self.bm25.normalized_top_score(q, float(s[i]))

    # ---- reading
    def encode(self, question, context):
        spans = [(m.start(), m.end()) for m in WORD.finditer(context)]
        q = [self.vocab.get(w.lower(), UNK_ID) for w in WORD.findall(question)]
        p = [self.vocab.get(context[a:b].lower(), UNK_ID) for a, b in spans]
        return q + [SEP_ID] + p, [0] * (len(q) + 1) + [1] * len(p), len(q) + 1, spans

    def read(self, question, context):
        """Returns (answer text, start char, end char, log-probability of the chosen start and end words, logits)."""
        ids, seg, off, spans = self.encode(question, context)
        lg = self.reader.logits(ids, seg)
        S, E = lg[off:, 0], lg[off:, 1]
        n = len(S)
        M = S[:, None] + E[None, :]
        ok = np.triu(np.ones((n, n), bool)) & ~np.triu(np.ones((n, n), bool), k=self.max_answer_words)
        i, j = np.unravel_index(np.where(ok, M, -1e9).argmax(), M.shape)
        conf = float(log_softmax(S)[i] + log_softmax(E)[j])
        return context[spans[i][0]:spans[j][1]], spans[i][0], spans[j][1], conf, lg

    # ---- refusal
    def confidence(self, bm25_norm, model_conf):
        r = self.bundle["refusal"]
        if r["signal"] == "BM25 score only":
            return bm25_norm
        if r["signal"] == "model confidence only":
            return model_conf
        x = (np.array([bm25_norm, model_conf]) - np.array(r["feature_mean"])) / np.array(r["feature_std"])
        return float(np.dot(x, r["lr_coef"]) + r["lr_intercept"])

    def answer(self, question):
        question = (question or "").strip()
        if not question or not WORD.search(question):
            return {"status": "invalid", "message": "Please type a question in Swahili."}
        question = question[:MAX_QUESTION_CHARS]
        pid, bm = self.retrieve(question)
        text, start, end, model_conf, _ = self.read(question, self.contexts[pid])
        conf = self.confidence(bm, model_conf)
        tau = self.bundle["refusal"]["threshold"]
        base = {"confidence": round(float(conf), 3), "threshold": round(float(tau), 3), "bm25_normalized": round(bm, 3), "model_log_probability": round(model_conf, 3)}
        if conf < tau:
            return {"status": "declined", "message": "I can only answer questions about horticulture (fruit, spice and vegetable farming) "
                    "that are covered by my farming manuals, and I am not confident this is one of them.", **base}
        return {"status": "answered", "answer": text, "answer_start": start, "answer_end": end, "passage_id": pid, "passage": self.contexts[pid], **base}
