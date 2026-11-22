\[!\[Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](https://colab.research.google.com/github/aparajitha0/sentiment-tracker/blob/main/notebooks/run\_app.ipynb)



\# 📊 InsightFeed: Social Media Sentiment Tracker



Upload any dataset of social media posts and get every post labelled

\*\*Positive, Negative or Neutral\*\*, with a pie chart, word clouds, short

post summaries and a downloadable CSV.



Built for CODEFORGE 3.0 HACKATHON at RNS Institute of Technology, Bangalore by \*\*Team Nova\*\*.



\## Features

\- \*\*Works on any dataset\*\*: finds the id, title and text columns on its own (CSV, Excel, JSON) and cleans the text (URLs, HTML, markdown, duplicates)

\- \*\*Two sentiment engines\*\*: RoBERTa (accurate, understands context) and VADER (fast baseline)

\- \*\*Dashboard\*\*: sentiment counts, pie chart and word clouds for each class

\- \*\*Automatic stopwords\*\*: hides words that appear in too many posts, so nothing is hardcoded for one dataset

\- \*\*Summaries\*\*: a free summary for every post, plus optional AI-written summaries using an Anthropic API key

\- \*\*Export\*\*: download `post\_id, sentiment, summary` as a CSV



\## How it works

Upload → clean (`clean\_data.py`) → label with RoBERTa or VADER → summarise (`summarizer.py`) → counts, pie chart, word clouds (`sentiment.py`) → download CSV



\## Model choice

We hand-labelled 51 posts from r/UofT and compared both models:



| Model   | Accuracy | Macro-F1 |

|---------|----------|----------|

| VADER   | 0.53     | 0.47     |

| RoBERTa | 0.80     | 0.70     |



VADER often called plain questions "positive" because of polite words like "thanks".

RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) reads the whole sentence, so it did much better.



\## Run it

\*\*Easiest: in Google Colab\*\*

1\. Click the "Open In Colab" button at the top.

2\. Go to Runtime → Change runtime type → T4 GPU.

3\. Click Runtime → Run all.

4\. Wait for the model to download, then click the public link shown in the output (`https://xxxx.gradio.live`).



\*\*On your own computer\*\*

```bash

git clone https://github.com/aparajitha0/sentiment-tracker.git

cd sentiment-tracker

pip install -r requirements.txt

python -c "from app import launch; launch()"

```



\## Try it with sample data

Upload `data/campus\_fest\_sample.csv` in the app to see how it works.



\## Datasets

\- `data/campus\_fest\_sample.csv`: simulated campus fest posts (included)

\- r/UofT Reddit posts: used during development, not included because they are scraped Reddit content





\## Project files

| File | What it does |

|---|---|

| `app.py` | The web dashboard (Gradio) |

| `clean\_data.py` | Loads and cleans any dataset into `post\_id, title, post` |

| `sentiment.py` | Sentiment labels, counts, pie chart, word clouds |

| `summarizer.py` | Short summary for each post |

| `notebooks/run\_app.ipynb` | Colab notebook that launches the app |



\## Limitations

\- Checked on only 51 hand-labelled posts by one person, so 80% is not an overall accuracy figure

\- Only 2 of those 51 posts were truly positive

\- English only; long posts are cut off at 512 tokens

\- Tested on one real dataset so far (r/UofT)



\## Next steps

\- Larger random validation set with a second person labelling

\- Test on more datasets

\- Deploy permanently on Hugging Face Spaces



\## Tech used

Python, pandas, Hugging Face Transformers, VADER, Gradio, WordCloud, matplotlib



\## Author

1. P Aparajitha, CSE-DS, RNSIT Bangalore · \[LinkedIn](https://www.linkedin.com/in/p-aparajitha)
2. Neha Bhat, CSE-DS, RNSIT Bangalore · \[LinkedIn](https://www.linkedin.com/in/neha-bhat-162455368)
3. Preksha R, CSE-DS, RNSIT Bangalore · \[LinkedIn](https://www.linkedin.com/in/prekshar)
4. Manasvi L G, CSE-DS, RNSIT Bangalore · \[LinkedIn](https://www.linkedin.com/in/manasvi-l-g-6700b43a5)



