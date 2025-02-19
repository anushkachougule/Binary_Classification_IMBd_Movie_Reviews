# IMDb Sentiment Analysis: Traditional ML, LSTM, and RoBERTa

## 📌 Overview  
This repository contains implementations for **sentiment analysis** on the **IMDb movie reviews dataset** using **three different approaches**:  

1. **Traditional Machine Learning Models** (Logistic Regression, SVM)  
2. **Deep Learning with LSTM** (using pre-trained GloVe embeddings)  
3. **Transformer-Based Fine-Tuning with RoBERTa**  

The objective is to compare different methods for classifying sentiment (positive/negative) in movie reviews and analyze their performance based on accuracy, training time, and resource requirements.  

---

## Repository Contents  

### Traditional Machine Learning Models  
- [`traditional_machine_learning_models.ipynb`](https://github.com/anushkachougule/Binary_Classification_IMBd_Movie_Reviews#:~:text=LSTM_binary_classification.ipynb) → Implements **Logistic Regression and SVM** with TF-IDF vectorization.  
- **Colab Link:**[ [Link]](https://drive.google.com/file/d/10A_HZzjuCJt6GdbxNcZS7Q2k9Xz2-SRX/view?usp=sharing)  

### LSTM-Based Sentiment Classification  
- [`LSTM_binary_classification.ipynb`](l[stm_model.ipynb](https://github.com/anushkachougule/Binary_Classification_IMBd_Movie_Reviews#:~:text=Traditional_machine_learning_models.ipynb)) → Implements **Bidirectional LSTM with GloVe embeddings**.  
- **Colab Link:** [[Link]  ](https://colab.research.google.com/drive/11WnWdFB4bcZE9oMTsKUSdyLDysNIxl4o?usp=sharing)

### Transformer-Based Sentiment Analysis (RoBERTa)  
- [`transformer_based_binary_classification.ipynb`]([roberta_finetuning.ipynb](https://github.com/anushkachougule/Binary_Classification_IMBd_Movie_Reviews#:~:text=Transformer_based_binary_classification.ipynb)) → Fine-tunes **RoBERTaForSequenceClassification** on the IMDb dataset.  
- **Colab Link:** [[Link] ](https://drive.google.com/file/d/1NfN77A_fi72TgIpQf_dG6BeNCe8rjVcZ/view?usp=sharing) 

---
## Dependencies & Setup
Before running the code locally, you need to install the required libraries:

Install general dependencies: **pip install transformers datasets torch torchtext scikit-learn numpy pandas matplotlib**

For LSTM training, install either TensorFlow or PyTorch: **pip install tensorflow or pip install torch**

For RoBERTa fine-tuning, install Hugging Face's transformers and datasets: **pip install transformers datasets accelerate**

If using Google Colab, enable GPU support by: Going to Runtime -> Clicking Change runtime type -> Selecting GPU


---
## 📊 Dataset  
The IMDb dataset can be accessed here:  
🔗 **[IMDb Dataset Source]([https://ai.stanford.edu/~amaas/data/sentiment/](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews))**  

Alternatively, you can load the dataset directly in **Google Colab** or **Hugging Face Datasets**:  

```python
from datasets import load_dataset
dataset = load_dataset("imdb")
