# OneZero Exercise

## Description
A Python data science project for the OneZero exercise.

## Project Structure
```
onezero_ex_yotam/
├── README.md
├── main.py
├── requirements.txt
├── create_index.py
├── evaluate_rag.py
├── qna_bot.py
├── build_rag.ipynb
├── docs/
│   ├── cards.md
│   └── securities.md
├── evaluation_data/
│   ├── questions.json
│   ├── chanks_recall_questions.json
│   └── results/
├── prompts/
└── vectors/
    ├── vector_store_data_md_spliter.json
    └── vector_store_data_md_spliter_chunk.json
```

## Installation
```bash
pip install -r requirements.txt
```
## create .env file with:

OPENAI_API_KEY = ....

## create index
```bash
python create_index.py
```
## Usage
```bash
streamlit run main.py
```
## Evaluation
```bash
python evaluate_rag.py
```
the results for now:
Average Ground Truth Score: 5.0
Average Correctness Score: 4.9
Average Full Recall: 1.0
Average Basic Recall: 1.0
Average Precision: 1.0
## Requirements
- Python 3.11+

