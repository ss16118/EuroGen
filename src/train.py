import os
from preprocess import *
from models import *
import tensorflow as tf
from generate_midi import *

if __name__ == '__main__':

    notes = get_notes()

    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = tf.keras.models.load_model("LSTM_model.h5")
    model.summary()

    nb_epochs = 3
    model.fit(network_input, network_output, epochs=nb_epochs, batch_size=64)
    model.save("LSTM_model.h5")

    generated_output = generate_notes(model, notes, network_input, n_vocab)
    create_midi(generated_output, "lstm_eurobeat_5")