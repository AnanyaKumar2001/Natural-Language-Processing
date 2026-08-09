# Natural Language Processing — Practical Notebooks

A hands-on walkthrough of classical NLP, from raw text to word embeddings. Each notebook is a
self-contained lesson and they are numbered so they can be followed in order: the early notebooks
build the text-preprocessing pipeline (tokenize → stem/lemmatize → remove stopwords), the middle
ones add linguistic annotation (POS tagging, NER), and the last three turn text into numeric
vectors (Bag of Words, TF-IDF, Word2Vec).

Repository: https://github.com/AnanyaKumar2001/Natural-Language-Processing

---

## Files in this folder

| File | What it covers |
| --- | --- |
| `01-Tokenization+Example+Using+NLTK.ipynb` | Splitting text into units with NLTK. Uses `sent_tokenize` (paragraph → sentences) and `word_tokenize`, `wordpunct_tokenize`, and `TreebankWordTokenizer` (sentences → words), comparing how each one treats punctuation. |
| `02-Stemming+And+Its+Types-+Text+Preprocessing.ipynb` | Reducing words to their stems. Compares `PorterStemmer`, `RegexpStemmer` (rule/regex based, e.g. stripping `ing$\|s$\|e$\|able$`), and `SnowballStemmer` (Porter2), including cases where Porter fails and Snowball does not (`fairly`, `sportingly`). |
| `03-Lemmatization-+Text+Preprocessing.ipynb` | `WordNetLemmatizer` — producing a valid dictionary word (lemma) instead of a truncated stem, and how the `pos` argument (`n`, `v`, `a`, `r`) changes the result. Notes where lemmatization is preferred: Q&A, chatbots, text summarization. |
| `04-Text+Preprocessing-Stopwords+With+NLTK.ipynb` | Full preprocessing pipeline on a real corpus — Dr. APJ Abdul Kalam's "three visions for India" speech. Downloads the NLTK `stopwords` corpus, filters stopwords, then runs the same text through Porter stemming, Snowball stemming, and WordNet lemmatization to contrast the outputs. |
| `05-Parts+Of+Speech+Tagging.ipynb` | POS tagging with `nltk.pos_tag` on the same Kalam speech, plus a reference table of Penn Treebank tags (`NN`, `JJ`, `VB`, `RB`, …). Requires the `averaged_perceptron_tagger` model. |
| `06-Named+Entity+Recognition.ipynb` | Named Entity Recognition on an Eiffel Tower sentence: tokenize → POS tag → `nltk.ne_chunk`, then `.draw()` to render the chunk tree. Lists the entity categories (person, location, date, time, money, organization, percent). Requires `maxent_ne_chunker` and `words`. |
| `07-Bag+Of+Words+Practical's.ipynb` | Bag of Words on the SMS Spam Collection dataset. Cleans each message with a regex, lowercases, removes stopwords, applies Porter stemming, then vectorizes with scikit-learn's `CountVectorizer` (`max_features=100`, `binary=True`) and repeats it with `ngram_range=(2,3)` to show bigram/trigram features. |
| `08-TF-IDF+Practical.ipynb` | The same spam corpus scored with `TfidfVectorizer` instead of raw counts, using WordNet lemmatization in the cleaning step, then an n-gram variant with `ngram_range=(2,2)`. Shows how TF-IDF down-weights terms that appear across many documents. |
| `09-Word2vec_Practical_Implementation.ipynb` | Dense word embeddings with `gensim`. Loads Google's pretrained `word2vec-google-news-300` vectors, inspects a 300-dim vector, runs `most_similar` and `similarity` queries, and demonstrates vector arithmetic — `king - man + woman`. |
| `.gitignore` | Excludes `venv/` from version control. |
| `venv/` | Local Python 3.11 (conda) environment used to run the notebooks. Not tracked by git. |

---

## Setup

The notebooks were run against the bundled `venv/` (Python 3.11, conda-managed). To recreate an
equivalent environment from scratch:

```bash
python -m venv venv
venv/Scripts/activate          # Windows
pip install nltk pandas numpy scikit-learn gensim ipykernel jupyter
```

Several NLTK data packages are downloaded from inside the notebooks:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
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
- **Notebooks 07–09 need extra packages.** The bundled `venv/` currently has `nltk` and `numpy` but
  **not** `pandas`, `scikit-learn`, or `gensim`. Install those before running the vectorization
  notebooks.
- **The SMS Spam Collection dataset is not in this folder.** Notebooks 07 and 08 read it from
  `smsspamcollection/SMSSpamCollection` and `SpamClassifier-master/smsspamcollection/SMSSpamCollection`
  respectively. Download the dataset (UCI SMS Spam Collection — a tab-separated `label \t message`
  file) and place it at those paths, or update the `pd.read_csv` path in each notebook.
- **Notebook 06 opens a GUI window.** `nltk.ne_chunk(...).draw()` renders the parse tree in a
  Tkinter window, so it will not display in a headless environment.
- **Notebook 09 downloads ~1.6 GB.** `api.load('word2vec-google-news-300')` fetches the pretrained
  Google News vectors on first run and caches them under `~/gensim-data`.

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
  └─ 09  Word2Vec ............... dense 300-dim vectors that capture meaning
```
