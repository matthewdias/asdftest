import math
import csv
import copy
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cluster import estimate_bandwidth, MeanShift
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec

#Initialize with coordinates of a place in a city.
centerCoordinates = [40.1891, -8.6321]

def np_haversine(lattitudeslongitudes1, lattitudeslongitudes2):
    radEarth = 6371
	lattitudes1 = lattitudeslongitudes1[:, 0]
    longitudes1 = lattitudeslongitudes1[:, 1]
    lattitudes2 = lattitudeslongitudes2[:, 0]
    longitudes2 = lattitudeslongitudes2[:, 1]
    lattitudes = np.abs(lattitudes1 - lattitudes2) * np.pi / 180
    longitudes = np.abs(longitudes1 - longitudes2) * np.pi / 180
    lattitudes1 = lattitudes1 * np.pi / 180
    lattitudes2 = lattitudes2 * np.pi / 180
    calc1 = np.sin(lattitudes / 2) * np.sin(lattitudes / 2) + np.cos(lattitudes1) * np.cos(lattitudes2) * np.sin(longitudes / 2) * np.sin(longitudes / 2)
    dist = 2 * np.arctan2(np.sqrt(calc1), np.sqrt(1 - calc1))
    return radEarth * dist

def tf_haversine(lattitudeslongitudes1, lattitudeslongitudes2):
    radEarth = 6371
	lattitudes1 = lattitudeslongitudes1[:, 0]
    longitudes1 = lattitudeslongitudes1[:, 1]
    lattitudes2 = lattitudeslongitudes2[:, 0]
    longitudes2 = lattitudeslongitudes2[:, 1]
    lattitudes = tf.abs(lattitudes1 - lattitudes2) * np.pi / 180
    longitudes = tf.abs(longitudes1 - longitudes2) * np.pi / 180
    lattitudes1 = lattitudes1 * np.pi / 180
    lattitudes2 = lattitudes2 * np.pi / 180
    calc1 = tf.sin(lattitudes / 2) * tf.sin(lattitudes / 2) + tf.cos(lattitudes1) * tf.cos(lattitudes2) * tf.sin(longitudes / 2) * tf.sin(longitudes / 2)
    dist = 2 * tf_atan2(tf.sqrt(calc1), tf.sqrt(1 - calc1))
    return radEarth * dist

def tf_atan2(y, x):
    angle = tf.select(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.select(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.select(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), np.nan * tf.zeros_like(x), angle)
    return angle

def get_clusters(coords):
    clusters = pd.DataFrame({'approx_lattitudesitudes': coords[:,0].round(4), 'approx_longitudesgitudes': coords[:,1].round(4)})
    clusters = clusters.drop_duplicates(['approx_lattitudesitudes', 'approx_longitudesgitudes'])
    clusters = clusters.as_matrix()
    bandwidth = estimate_bandwidth(clusters, quantile=0.0002)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(clusters)
    return ms.cluster_centers_

def density_map(lattitudesitudes, longitudesgitudes, center=centerCoordinates, bins=1000, radius=0.1):
    cmap = copy.copy(plt.cm.jet)
    cmap.set_bad((0,0,0))
    histogram_range = [[center[1] - radius, center[1] + radius], [center[0] - radius, center[0] + radius]]
    plt.figure(figsize=(5,5))
    plt.hist2d(longitudesgitudes, lattitudesitudes, bins=bins, norm=LogNorm(), cmap=cmap, range=histogram_range)
    plt.grid('off')
    plt.axis('off')
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.tight_layout()
    plt.show()    

def plot_embeddings(embeddings):
    N = len(embeddings)
    cols = 2
    rows = int(math.ceil(N / float(cols)))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(8,7))
    for i, embedding in enumerate(embeddings):
        ax = fig.add_subplot(gs[i])
        weights = embedding[1].get_weights()[0]
        names = range(weights.shape[0])
        tsne = TSNE(n_components=2, random_state=0)
        tsne = tsne.fit_transform(weights)
        x, y = tsne[:,0], tsne[:,1]
        scatter = ax.scatter(x, y, alpha=0.7, c=names, s=40, cmap="jet")
        fig.colorbar(scatter, ax=ax)
        for i, name in enumerate(names):
            ax.annotate(name, (x[i], y[i]), size=6)
        x_delta = x.max() - x.min()
        x_margin = x_delta / 10
        y_delta = y.max() - y.min()
        y_margin = y_delta / 10
        ax.set_xlim(x.min()-x_margin, x.max()+x_margin)
        ax.set_ylim(y.min()-y_margin, y.max()+y_margin)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(embedding[0])
        fig.tight_layout()
    plt.show()

def export_answers(model, competition_test, filename='answers.csv'):
    from code.training import process_features
    predictions = model.predict(process_features(competition_test))
    answer_csv = open(filename, 'w')
    answer_csv = csv.writer(answer_csv, quoting=csv.QUOTE_NONNUMERIC)
    answer_csv.writerow(['TRIP_ID', 'LATITUDE', 'LONGITUDE'])
    for index, (lattitudesitude, longitudesgitude) in enumerate(predictions):
        answer_csv.writerow([competition_test['TRIP_ID'][index], lattitudesitude, longitudesgitude])
