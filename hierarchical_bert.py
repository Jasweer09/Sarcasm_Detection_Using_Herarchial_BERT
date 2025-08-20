import tensorflow as tf
from tensorflow import keras
from keras.utils import register_keras_serializable
@register_keras_serializable()
class HierarchicalBERT(tf.keras.Model):
  def __init__(self, bert_model, lstm_units, cnn_filters, dense_units):
    super(HierarchicalBERT, self).__init__()
    self.bert = bert_model

    #sentence encoding layer
    self.dense_sentense = tf.keras.layers.Dense(768, activation='relu')

    #Context Summarization layer
    #adding or pooling above individual vectors into summarized context layer
    self.mean_pooling = tf.keras.layers.GlobalAveragePooling1D()

    #Context Encoder Layer
    #Here we are using the LSTM for capturing the temporal dependencies and context of summaried data from above from both sides
    self.bilstm_encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences = True))

    #CNN Layer
    #it will extract the local features
    self.conv = tf.keras.layers.Conv1D(cnn_filters, 2, activation='relu')
    self.pool = tf.keras.layers.GlobalMaxPooling1D()

    #Fully connected layer
    self.dense = tf.keras.layers.Dense(dense_units, activation='relu')
    #Output Layer
    self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
  def call(self, inputs):
    #BERT embedding
    bert_output = self.bert(inputs)[0]
    #sentence encoding layer
    sentence_encoded = self.dense_sentense(bert_output)

    #context summarization layer
    context_summarized = self.mean_pooling(sentence_encoded)

    #expand the dimension
    context_summarized = tf.expand_dims(context_summarized, 1)

    #context encoder layer
    context_encoded = self.bilstm_encoder(context_summarized)

    #squeezing the dimension
    context_encoded_squeezed = tf.squeeze(context_encoded, axis = 1)

    #adding the channel dimension as required input shapeby convlayer
    context_encoded_expanded = tf.expand_dims(context_encoded_squeezed, axis = -1)
    #CNN layer
    cnn_output = self.conv(context_encoded_expanded)
    cnn_output = self.pool(cnn_output)
    #Fully contected layer
    dense_output = self.dense(cnn_output)
    #output layer
    output = self.output_layer(dense_output)

    return output
