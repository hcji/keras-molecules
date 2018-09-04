from __future__ import print_function

import argparse
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import h5py
import numpy as np
from molecules.model import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

NUM_EPOCHS = 100
BATCH_SIZE = 10
LATENT_DIM = 128
RANDOM_SEED = 123

np.random.seed(RANDOM_SEED)#args.random_seed)


data_train, data_test, charset = load_dataset('./data/processed.h5')
model = MoleculeVAE()
#model.load(charset, args.model, latent_rep_size = args.latent_dim)
model.create(charset, latent_rep_size=LATENT_DIM)

checkpointer = ModelCheckpoint(filepath = './test_models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                               verbose = 1,
                               save_best_only = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 3,
                              min_lr = 0.0001)

data_train = data_train[:1]
model.autoencoder.fit(
    data_train,
    data_train,
    shuffle = True,
    nb_epoch = NUM_EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks = [checkpointer, reduce_lr],
    validation_data = (data_test, data_test)
)
