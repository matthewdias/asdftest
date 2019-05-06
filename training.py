import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adagrad
from keras import backend as K
from keras.layers.embeddings import Embedding
import keras
from keras.callbacks import ModelCheckpoint
from code.utils import tf_haversine
from code.data import load_data
from code.utils import get_clusters

#Starts a new Tensorflow session.
def start_new_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, graph=tf.get_default_graph())
    K.tensorflow_backend.set_session(session)

#Returns a list with the first k and last k GPS coordinates from the given trip.
def get_k_first_last(coordinates):
    k = 5
    listCoordinates = [coordinates[0] for i in range(2*k)]
    setOfCoordinates = len(coordinates)
    if setOfCoordinates < 2*k:
        listCoordinates[-setOfCoordinates:] = coordinates
    else:
        listCoordinates[:k] = coordinates[:k]
        listCoordinates[-k:] = coordinates[-k:]
    listCoordinates = np.row_stack(listCoordinates)
    return np.array(listCoordinates).flatten()

#Process the features required by our model from the given dataframe.
def process_features(df):
    coordinates = np.row_stack(df['POLYLINE'].apply(get_k_first_last))
    latitudes = coordinates[:,::2]
    coordinates[:,::2] = scale(latitudes)
    longitudes = coordinates[:,1::2]
    coordinates[:,1::2] = scale(longitudes)    
    return [df['QUARTER_HOUR'].as_matrix(), df['DAY_OF_WEEK'].as_matrix(), df['WEEK_OF_YEAR'].as_matrix(), df['ORIGIN_CALL_ENCODED'].as_matrix(), df['TAXI_ID_ENCODED'].as_matrix(), df['ORIGIN_STAND_ENCODED'].as_matrix(), coordinates]

#Creates all the layers for our neural network model.
def create_model(metadata, clusters):
    output_dim = 10
    max_length = 1

    #Quarter hour of the day embedding layer
    equarterhour = Sequential()
    equarterhour.add(Embedding(metadata['n_quarter_hours'], output_dim, input_length = max_length))
    equarterhour.add(Reshape((output_dim,)))

    #Day of the week embedding layer
    edayofweek = Sequential()
    edayofweek.add(Embedding(metadata['n_days_per_week'], output_dim, input_length = max_length))
    edayofweek.add(Reshape((output_dim,)))

    #Week of the year embedding layer
    eweekofyear = Sequential()
    eweekofyear.add(Embedding(metadata['n_weeks_per_year'], output_dim, input_length = max_length))
    eweekofyear.add(Reshape((output_dim,)))

    #Client ID embedding layer
    eclientids = Sequential()
    eclientids.add(Embedding(metadata['n_client_ids'], output_dim, input_length = max_length))
    eclientids.add(Reshape((output_dim,)))

    #Taxi ID embedding layer
    etaxiids = Sequential()
    etaxiids.add(Embedding(metadata['n_taxi_ids'], output_dim, input_length = max_length))
    etaxiids.add(Reshape((output_dim,)))

    #Taxi stand ID embedding layer
    estandids = Sequential()
    estandids.add(Embedding(metadata['n_stand_ids'], output_dim, input_length = max_length))
    estandids.add(Reshape((output_dim,)))
    
    #Coordinates (k first lattitude or longitude and k last lattitude or longitude, therefore we have total 2k values).
    coordinates = Sequential()
    coordinates.add(Dense(1, input_dim=20, init='normal'))

    #Merge all the inputs/features into a single input layer in our model.
    model = Sequential()
    model.add(Merge([equarterhour, edayofweek, eweekofyear, eclientids, etaxiids, estandids, setOfCoordinates], mode='concat'))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(len(clusters)))
    model.add(Activation('softmax'))
    cast_clusters = K.cast_to_floatx(clusters)
    def destination(probabilities):
        return tf.matmul(probabilities, cast_clusters)
    model.add(Activation(destination))

    # Compile the model
    optimizer = SGD(lr=0.01, momentum=0.9, clipvalue=1.)
    model.compile(loss=tf_haversine, optimizer=optimizer)
    
    return model

#Runs the complete training process.
def full_train(n_epochs=100, batch_size=200, save_prefix=None):
    #Load the given data set.
    print("Please wait while we load the data...")
    data = load_data()
    #Estimate the allClusters.
    print("Cluster computation...")
    allClusters = get_clusters(data.train_labels)    
    callbacks = []
    if save_prefix is not None:
        #Save the model's intermediate weights to disk after each epoch.
        file_path="cache/%s-{epoch:03d}-{val_loss:.4f}.hdf5" % save_prefix
        callbacks.append(ModelCheckpoint(file_path, monitor='val_loss', mode='min', save_weights_only=True, verbose=1))
    #Create the model.
    print("Creating model...")
    start_new_session()
    model = create_model(data.metadata, clusters)
    #Run the training algorithm and find the fit.
    print("Start training...")
    history = model.fit(
        process_features(data.train), data.train_labels,
        nb_epoch=n_epochs, batch_size=batch_size,
        validation_data=(process_features(data.validation), data.validation_labels),
        callbacks=callbacks)
    print(model.summary)
    if save_prefix is not None:
        #Save the training history to disk to use for every epoch.
        file_path = 'cache/%s-history.pickle' % save_prefix
        with open(file_path, 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return history
