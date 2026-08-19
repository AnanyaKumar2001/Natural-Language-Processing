# Natural Language Processing — Practical Notebooks

A hands-on walkthrough of classical NLP, from raw text to word embeddings to a working spam
classifier. Each notebook is a self-contained lesson and they are numbered so they can be followed
in order: the early notebooks build the text-preprocessing pipeline (tokenize → stem/lemmatize →
remove stopwords), the middle ones add linguistic annotation (POS tagging, NER), the next three turn
text into numeric vectors (Bag of Words, TF-IDF, Word2Vec), and the last three put all of it
together into end-to-end SMS spam/ham classification projects with scikit-learn.

Repository: https://github.com/AnanyaKumar2001/Natural-Language-Processing

---

## Files in this folder

### Notebooks — text preprocessing (01–06)

| File | What it covers |
| --- | --- |
| `01-Tokenization+Example+Using+NLTK.ipynb` | Splitting text into units with NLTK. Uses `sent_tokenize` (paragraph → sentences) and `word_tokenize`, `wordpunct_tokenize`, and `TreebankWordTokenizer` (sentences → words), comparing how each one treats punctuation. |
| `02-Stemming+And+Its+Types-+Text+Preprocessing.ipynb` | Reducing words to their stems. Compares `PorterStemmer`, `RegexpStemmer` (rule/regex based, e.g. stripping `ing$\|s$\|e$\|able$`), and `SnowballStemmer` (Porter2), including cases where Porter fails and Snowball does not (`fairly`, `sportingly`). |
| `03-Lemmatization-+Text+Preprocessing.ipynb` | `WordNetLemmatizer` — producing a valid dictionary word (lemma) instead of a truncated stem, and how the `pos` argument (`n`, `v`, `a`, `r`) changes the result. Notes where lemmatization is preferred: Q&A, chatbots, text summarization. |
| `04-Text+Preprocessing-Stopwords+With+NLTK.ipynb` | Full preprocessing pipeline on a real corpus — Dr. APJ Abdul Kalam's "three visions for India" speech. Downloads the NLTK `stopwords` corpus, filters stopwords, then runs the same text through Porter stemming, Snowball stemming, and WordNet lemmatization to contrast the outputs. |
| `05-Parts+Of+Speech+Tagging.ipynb` | POS tagging with `nltk.pos_tag` on the same Kalam speech, plus a reference table of Penn Treebank tags (`NN`, `JJ`, `VB`, `RB`, …). Requires the `averaged_perceptron_tagger` model. |
| `06-Named+Entity+Recognition.ipynb` | Named Entity Recognition on an Eiffel Tower sentence: tokenize → POS tag → `nltk.ne_chunk`, then `.draw()` to render the chunk tree. Lists the entity categories (person, location, date, time, money, organization, percent). Requires `maxent_ne_chunker` and `words`. |

### Notebooks — text to vectors (07–09)

| File | What it covers |
| --- | --- |
| `07-Bag+Of+Words+Practical's.ipynb` | Bag of Words on the SMS Spam Collection dataset. Cleans each message with a regex, lowercases, removes stopwords, applies Porter stemming, then vectorizes with scikit-learn's `CountVectorizer` (`max_features=100`, `binary=True`) and repeats it with `ngram_range=(2,3)` to show bigram/trigram features. |
| `08-TF-IDF+Practical.ipynb` | The same spam corpus scored with `TfidfVectorizer` instead of raw counts, using WordNet lemmatization in the cleaning step, then an n-gram variant with `ngram_range=(2,2)`. Shows how TF-IDF down-weights terms that appear across many documents. |
| `09-Word2vec_Practical_Implementation.ipynb` | Dense word embeddings with `gensim`. Loads Google's pretrained `word2vec-google-news-300` vectors, inspects a 300-dim vector, runs `most_similar` and `similarity` queries, and demonstrates vector arithmetic — `king - man + woman`. |

### Notebooks — spam/ham classification projects (10–12)

| File | What it covers |
| --- | --- |
| `10-Spam Ham Classification Project Using TF-IDF And ML.ipynb` | First end-to-end classifier. Clean and Porter-stem all 5,572 SMS messages, vectorize the **whole** corpus with `CountVectorizer(max_features=2500, ngram_range=(1,2))`, build the label vector with `pd.get_dummies`, then `train_test_split` and fit a `MultinomialNB`. Reports ~98.1% accuracy plus a full `classification_report`. Despite the filename, this notebook uses Bag of Words, not TF-IDF. |
| `11-Spam Ham Classification Project Using BOW And TFIDF And ML.ipynb` | The corrected version of notebook 10, and the one to prefer. Splits **before** vectorizing, so the vectorizer is `fit_transform`-ed on train and only `transform`-ed on test — no data leakage from the test set into the vocabulary. Runs the same pipeline twice, once with `CountVectorizer` (~98.3%) and once with `TfidfVectorizer` (~97.7%), both `max_features=2500, ngram_range=(1,2)` feeding a `MultinomialNB`. |
| `12-Spam Ham Projects Using Word2vec,AvgWord2vec.ipynb` | Embedding-based classifier. Lemmatizes the messages, tokenizes with `gensim.utils.simple_preprocess`, trains a `Word2Vec` model **from scratch on the SMS corpus itself** (the pretrained Google vectors are only loaded at the top for comparison), then builds one feature vector per message by averaging its word vectors (AvgWord2Vec). Assembles the vectors into a DataFrame, drops the messages that ended up empty, and fits a `RandomForestClassifier` — ~99.7% accuracy on the held-out split. |

### Data and supporting files

| File | What it is |
| --- | --- |
| `SMSSpamCollection.txt` | The UCI SMS Spam Collection v.1 — 5,574 tab-separated `label \t message` lines, ham or spam (pandas reads 5,572 rows, because its default quoting merges a few lines that contain a `"`). The corpus used by notebooks 07, 08, 10, 11, and 12. |
| `readme_smsspamcollection.md` | The dataset's own README: how the corpus was compiled (Grumbletext, the NUS SMS Corpus, Caroline Tag's PhD thesis, SMS Spam Corpus v.0.1 Big), its statistics, usage notes, and the papers to cite when using it. |
| `all_kindle_review.csv` | Amazon Kindle Store product reviews (~12,000 rows) with `rating`, `reviewText`, `summary`, `helpful`, and reviewer metadata. A sentiment-analysis dataset staged for future work — no notebook in this repo reads it yet. |
| `.gitignore` | Excludes `venv/` from version control. |
| `venv/` | Local Python 3.11 (conda) environment used to run the notebooks. Not tracked by git. |

---

## Setup

The notebooks were run against the bundled `venv/` (Python 3.11, conda-managed). To recreate an
equivalent environment from scratch:

```bash
python -m venv venv
venv/Scripts/activate          # Windows
pip install nltk pandas numpy scikit-learn gensim tqdm ipykernel jupyter
```

Several NLTK data packages are downloaded from inside the notebooks:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

Then launch Jupyter and open the notebooks in numeric order:

```bash
jupyter notebook
```

---

## Notes before running

- **Notebooks 01–06 run as-is.** They only need `nltk` plus the downloaded corpora, all of which
  the bundled `venv/` already has.
- **Notebooks 07–12 need extra packages.** The bundled `venv/` currently has `nltk`, `numpy`, and
  `tqdm` but **not** `pandas`, `scikit-learn`, or `gensim`. Install those before running the
  vectorization and classification notebooks.
- **The dataset is now in this folder, but the notebook paths do not point at it.**
  `SMSSpamCollection.txt` sits at the repository root, while notebooks 07, 10, 11, and 12 read
  `smsspamcollection/SMSSpamCollection` and notebook 08 reads
  `SpamClassifier-master/smsspamcollection/SMSSpamCollection`. Either change the `pd.read_csv` path
  to `'SMSSpamCollection.txt'`, or create the expected folder:

  ```bash
  mkdir -p smsspamcollection
  cp SMSSpamCollection.txt smsspamcollection/SMSSpamCollection
  ```

- **Notebook 06 opens a GUI window.** `nltk.ne_chunk(...).draw()` renders the parse tree in a
  Tkinter window, so it will not display in a headless environment.
- **Notebooks 09 and 12 download ~1.6 GB.** `api.load('word2vec-google-news-300')` fetches the
  pretrained Google News vectors on first run and caches them under `~/gensim-data`. Notebook 12
  loads them only to show a sample vector — the classifier it builds uses a Word2Vec model trained
  on the SMS corpus, so that cell can be skipped if the download is not wanted.
- **Notebook 12 uses `DataFrame.append`,** which was removed in pandas 2.0. On a modern pandas,
  replace the row-by-row accumulation loop with a single `pd.DataFrame(np.vstack(X))`.
- **Notebooks 10–12 were executed on Python 3.8.5,** not the bundled 3.11 environment; the stored
  outputs come from that run.
- **Accuracies vary run to run.** None of the `train_test_split` calls set `random_state`, so the
  numbers below shift by a few tenths of a percent on each execution.

---

## Results at a glance

| Notebook | Features | Model | Accuracy |
| --- | --- | --- | --- |
| 10 | BOW, 2500 features, 1–2 grams (fit on the full corpus) | MultinomialNB | 0.981 |
| 11 | BOW, 2500 features, 1–2 grams (fit on train only) | MultinomialNB | 0.983 |
| 11 | TF-IDF, 2500 features, 1–2 grams (fit on train only) | MultinomialNB | 0.977 |
| 12 | AvgWord2Vec, 100-dim, trained on the corpus | RandomForestClassifier | 0.997 |

---

## Concept map

```
Raw text
  │
  ├─ 01  Tokenization ........... text → sentences → words
  │
  ├─ 02  Stemming ............... words → stems (fast, may not be real words)
  ├─ 03  Lemmatization .......... words → lemmas (slower, always valid words)
  ├─ 04  Stopword removal ....... drop low-signal words, then stem/lemmatize
  │
  ├─ 05  POS tagging ............ label each word's grammatical role
  ├─ 06  NER ................... extract people, places, dates, organizations
  │
  ├─ 07  Bag of Words ........... sparse counts / binary occurrence + n-grams
  ├─ 08  TF-IDF ................. sparse counts weighted by document rarity
  ├─ 09  Word2Vec ............... dense 300-dim vectors that capture meaning
  │
  ├─ 10  BOW + Naive Bayes .............. first end-to-end classifier
  ├─ 11  BOW / TF-IDF + Naive Bayes ..... same, split before vectorizing
  └─ 12  AvgWord2Vec + Random Forest .... one dense vector per message
```
