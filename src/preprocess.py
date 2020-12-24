import glob
import os
import pickle

import numpy
from music21 import converter, instrument, note, chord
from tensorflow.python.keras.utils import np_utils

from src.utils import progress_bar
from constants import *


def get_notes():

    # If all the notes have already been collected and saved
    if os.path.exists(NOTES_PATH):
        print("Notes already collected...")
        with open(NOTES_PATH, "rb") as file:
            notes = pickle.load(file)
        print("All notes loaded from {}".format(NOTES_PATH))
        return notes

    all_midis = glob.glob("../midi/*.mid")
    num_of_files = len(all_midis)

    notes = []
    print("Retrieving all notes from training files...")
    for i, file in enumerate(all_midis):
        midi = converter.parse(file)

        parts = instrument.partitionByInstrument(midi)

        if parts:  # File has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:      # File has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        progress_bar(i + 1, num_of_files, "Current file", os.path.basename(file))

    # Save all the notes to a pickle file
    with open('../data/notes', 'wb') as file:
        pickle.dump(notes, file)
        print("\nAll notes saved to {}".format(file.name))


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitch_names = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output
