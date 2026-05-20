# Ted Talk Recommendation System

![Repository Language](https://img.shields.io/badge/language-Jupyter%20Notebook-blue)
![License](https://img.shields.io/github/license/anbuchezhiyan2005/Ted-Talk-Recommendation-System)
![Stars](https://img.shields.io/github/stars/anbuchezhiyan2005/Ted-Talk-Recommendation-System?style=social)

## Overview

The **Ted Talk Recommendation System** is a machine learning project designed to recommend TED Talks tailored to user preferences. Leveraging the data-driven capabilities of Jupyter Notebook and Python, this project analyzes a large collection of TED Talk metadata to provide personalized suggestions for viewers seeking inspiration, education, or entertainment.

## Features

- 📚 **Content-based filtering** for recommending talks based on topics and user interests.
- 🤖 **Natural Language Processing (NLP)** techniques for analyzing talk transcripts and descriptions.
- 📊 **Data visualizations** to understand talk trends and categories.
- 🔎 **Search and filter** talks by tags, speakers, or popularity.

## Tech Stack

- **Jupyter Notebook** (Main development environment, 99.6% of codebase)
- **Python** (Core scripting, 0.4%)
- **Pandas**, **Scikit-learn**, **NLTK/Spacy** (for NLP)
- **Matplotlib**, **Seaborn** (for visualization)

## Dataset

The project utilizes a TED Talks dataset containing fields such as:

- Title
- Speaker
- Tags
- Transcript/Text
- Duration
- Published Date
- View count

*(Dataset source: [Kaggle TED Talks Dataset](https://www.kaggle.com/datasets/rounakbanik/ted-talks), or specify your actual data source.)*

## How It Works

1. **Data Preprocessing:**  
   Cleaning and structuring TED Talk data for modeling.

2. **Feature Extraction:**  
   Applying NLP techniques to understand topics and sentiments.

3. **Model Training:**  
   Building a content-based recommendation engine using TF-IDF and cosine similarity.

4. **Evaluation:**  
   Assessing performance with metrics like precision and recall.

5. **Visualization:**  
   Creating plots to explore trends in talk topics, speakers, and popularity.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required packages listed in `requirements.txt`
  
### Installation

```bash
git clone https://github.com/anbuchezhiyan2005/Ted-Talk-Recommendation-System.git
cd Ted-Talk-Recommendation-System
pip install -r requirements.txt
```

### Usage

1. Open `ted_talk_recommendation.ipynb` in Jupyter Notebook.
2. Run the notebook cells sequentially.
3. Follow on-screen instructions to input your preferences or select examples.

### Example

```python
# Sample usage in notebook
user_input = "motivation, leadership, education"
recommended_talks = get_recommendations(user_input)
print(recommended_talks)
```

## Project Structure

```
Ted-Talk-Recommendation-System/
│
├── data/                # Dataset files
├── ted_talk_recommendation.ipynb  # Main Notebook
├── requirements.txt     # Dependencies
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for enhancements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [TED Talks](https://www.ted.com/)
- [Kaggle TED Talks Dataset](https://www.kaggle.com/datasets/rounakbanik/ted-talks)
- Open-source Python libraries: pandas, scikit-learn, NLTK, etc.

---

> *"Ideas worth spreading."* — TED
