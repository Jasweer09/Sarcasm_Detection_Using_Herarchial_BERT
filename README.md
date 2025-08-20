# Sarcasm_Detection_Using_Herarchial_BERT

This project implements a sarcasm detection system using **deep learning with a lightweight neural network** (Embedding + Pooling + Dense), instead of using **large pre-trained transformer models like BERT**.  

---

## 🔑 Difference from Normal BERT  
- **BERT (Bidirectional Encoder Representations from Transformers):**  
  - Pre-trained on massive corpora (BooksCorpus + Wikipedia).  
  - Uses **transformer encoders** with multi-head attention to capture **long-range dependencies** and **bidirectional context**.  
  - Each word’s meaning is dynamically influenced by surrounding words.  
  - Requires only **fine-tuning** for downstream tasks like sarcasm detection.  
  - Computationally heavy but highly accurate for nuanced text understanding.  

- **Our Model (Embedding + GlobalAveragePooling + Dense):**  
  - **Learns embeddings from scratch** on the sarcasm dataset.  
  - Uses **GlobalAveragePooling1D** to average embeddings across tokens, treating all words with equal importance (no attention).  
  - Architecture is **shallow**: no transformers, no pre-training.  
  - Lightweight, faster, and easy to train with limited resources.  
  - Lacks contextual depth → cannot distinguish subtle sarcastic cues that rely on tone or long-range semantics.  

👉 **Summary:** Unlike BERT, our model is a **bag-of-embeddings classifier**. BERT encodes context-aware word representations, while our model relies only on averaged static embeddings. This trade-off makes our system simple and efficient, but less context-sensitive.  

---

## 1. Importing Libraries  
- TensorFlow/Keras for modeling.  
- NumPy, Pandas for data handling.  
- Matplotlib for plotting.  

---

## 2. Data Loading  
- Dataset: **`sarcasm.json`**.  
- Fields: `headline` (text), `is_sarcastic` (label), `article_link` (unused).  
- Loaded into Pandas DataFrame.  

---

## 3. Data Preparation  
- Extract headlines → input sentences.  
- Extract labels → supervised targets.  
- Convert to NumPy arrays.  
- Split into **training/validation sets**.  

---

## 4. Tokenization and Padding  
- Keras `Tokenizer` with fixed vocabulary size + OOV token.  
- Text → integer sequences.  
- `pad_sequences` ensures uniform input length (padding/truncation).  

---

## 5. Model Architecture  

+-------------------------+
|        Input Layer      |
|  Tokenized Text (e.g.,  |
|  input_ids from BERT    |
|  Tokenizer, shape:      |
|  (batch_size, seq_len)  |
+-------------------------+
            |
            v
+-------------------------+
|     BERT Embedding      |
| TFBertModel (pre-trained|
| 'bert-base-uncased')    |
| Output: (batch_size,    |
| seq_len, 768)           |
+-------------------------+
            |
            v
+-------------------------+
|  Sentence Encoding Layer|
| Dense(768, activation=  |
| 'relu')                 |
| Processes BERT output   |
| to encode sentences.    |
+-------------------------+
            |
            v
+-------------------------+
|Context Summarization    |
| GlobalAveragePooling1D()|
| Averages across sequence|
| to summarize context.   |
| Output: (batch_size, 768)|
+-------------------------+
            |
            v
+-------------------------+
| Dimension Expansion     |
| tf.expand_dims(..., axis=1)|
| Prepares for LSTM:     |
| (batch_size, 1, 768)    |
+-------------------------+
            |
            v
+-------------------------+
|   Context Encoder Layer |
| Bidirectional(          |
| LSTM(128, return_seq=True))|
| Captures temporal       |
| dependencies bidirection-|
| ally.                   |
| Output: (batch_size, 1, |
| 256) [bi-directional]   |
+-------------------------+
            |
            v
+-------------------------+
| Dimension Squeeze       |
| tf.squeeze(..., axis=1) |
| Removes singleton dim:  |
| (batch_size, 256)       |
+-------------------------+
            |
            v
+-------------------------+
| Dimension Expansion for |
| CNN                     |
| tf.expand_dims(...,     |
| axis=-1)                |
| Adds channel:           |
| (batch_size, 256, 1)    |
+-------------------------+
            |
            v
+-------------------------+
|      CNN Layer          |
| Conv1D(filters=64,      |
| kernel_size=2,          |
| activation='relu')      |
| Extracts local features.|
| Followed by:            |
| GlobalMaxPooling1D()    |
| Output: (batch_size, 64)|
+-------------------------+
            |
            v
+-------------------------+
| Fully Connected Layer   |
| Dense(32, activation=   |
| 'relu')                 |
| Learns higher-level     |
| representations.        |
+-------------------------+
            |
            v
+-------------------------+
|       Output Layer      |
| Dense(1, activation=    |
| 'sigmoid')              |
| Binary classification   |
| (Sarcasm: 1, Non: 0)    |
+-------------------------+

- **Embedding Layer**: Creates dense representations of words.  
- **GlobalAveragePooling1D**: Compresses entire sentence into one vector by averaging.  
- **Dense (ReLU)**: Learns intermediate features.  
- **Dense (Sigmoid)**: Binary classifier (sarcastic vs. non-sarcastic).  

---

## 6. Compilation  
- Loss: **binary_crossentropy**.  
- Optimizer: **adam**.  
- Metric: **accuracy**.  

---

## 7. Training  
- Trained for multiple epochs.  
- Training and validation accuracy/loss tracked.  

---

## 8. Evaluation  
- Accuracy & loss curves plotted using Matplotlib.  
- Useful for diagnosing **overfitting/underfitting**.  

---

## 9. Results  
- Achieves reasonable accuracy on sarcasm detection.  
- Embedding + pooling provides efficiency but limited depth.  

---

## 10. Key Takeaways  
- Tokenization + padding are critical preprocessing steps.  
- Embeddings learn semantic relationships.  
- Even simple models can work well for binary classification.  
- Visualization of training history is essential for debugging learning patterns.  

---

## 11. Next Steps (Possible Improvements)  
- Replace pooling with **LSTM/GRU** to model sequential dependencies.  
- Use **pre-trained embeddings (Word2Vec, GloVe)** for richer word knowledge.  
- Upgrade to **BERT/transformer models** for contextual understanding.  
- Perform **hyperparameter tuning** and **expand dataset** for better generalization.  

---

# ✅ Final Note  
This project differs from **BERT-based sarcasm detection** in that it:  
- Does **not** use pre-trained contextual embeddings.  
- Does **not** apply attention or transformer encoders.  
- Relies on **average pooling of embeddings**, treating all words equally.  
- Focuses on **speed and simplicity** rather than deep contextual nuance.  

This makes the model **fast and resource-efficient**, but less powerful than BERT when detecting sarcasm that depends on subtle context or long-range word interactions.  
