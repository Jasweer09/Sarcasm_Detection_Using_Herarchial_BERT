from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import uvicorn
from tensorflow import keras
from hierarchical_bert import HierarchicalBERT

# ----------- Paths -----------
MODEL_PATH = "my_nlp_model.keras"
TOKENIZER_PATH = "tokenizer"

# ----------- Load Tokenizer & BERT Model -----------
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
bert_model = TFBertModel.from_pretrained("bert-base-uncased")  # make sure same BERT used during training

# ----------- Load HierarchicalBERT Model -----------
model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "HierarchicalBERT": lambda **kwargs: HierarchicalBERT(
            bert_model=bert_model,
            lstm_units=64,       # use same values as during training
            cnn_filters=32,
            dense_units=128,
            **kwargs
        )
    }
)

# ----------- FastAPI App -----------
app = FastAPI(title="Sarcasm Detection API")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sarcasm Detection API is running!"}

@app.post("/predict")
def predict(input_data: TextInput):
    # Tokenize input
    inputs = tokenizer(
        input_data.text,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # ---- Model Inference ----
    probs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs.get("token_type_ids")  # optional for some tokenizers
    ).numpy()[0][0]

    label = "Sarcasm" if probs > 0.5 else "Not Sarcasm"

    return {
        "text": input_data.text,
        "prediction": label,
        "confidence": float(probs if probs > 0.5 else 1 - probs)
    }

# ----------- Run App -----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
