import numpy as np
import itertools
import sys
from config.constants import Similarity_Methods

MAX_DISTANCE = 9999

class Hierarchy:
    def __init__(self, points, classifications, method=Similarity_Methods.CENTROID):
        self.method = method
        self.points = points
        self.classifications = classifications
        self.centroids = None
        self.clusters = None
        self.cluster_classifications = None
        self.cluster_mapping = np.arange(0, len(points))
        # Set up distance matrix
        self.distances = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            self.distances[i][i] = MAX_DISTANCE
            for j in range(len(points)):
                if i < j:
                    d = self.euclidean_distance(points[i], points[j])
                    self.distances[i][j] = d
                    self.distances[j][i] = d

    def run(self, cluster_limit):
        while len(np.unique(self.cluster_mapping)) > cluster_limit:
            print("\tn =", len(np.unique(self.cluster_mapping)))
            clusters = np.unique(self.cluster_mapping)
            self.calculate_centroids(self.cluster_mapping)
            self.merge_closest_clusters()
        self.calculate_cluster_classification()
        classes = []
        for cluster in self.clusters:
            points = np.where(self.cluster_mapping == self.cluster_mapping[cluster])[0]
            classes.append(points.tolist())
        return classes

    def merge_closest_clusters(self):
        index = np.argmin(self.distances)
        p1 = int(index / self.distances.shape[0])
        p2 = index % self.distances.shape[1]
        p2_friends = np.where(self.cluster_mapping == self.cluster_mapping[p2])[0]

        # Convert all Cluster B elems to Cluster A
        for elem in p2_friends:
            self.cluster_mapping[elem] = self.cluster_mapping[p1]
            for p in range(len(self.points)):
                self.distances[elem][p] = MAX_DISTANCE
                self.distances[p][elem] = MAX_DISTANCE
        # p1 is the cluster's ambassador. Remove distances from all other cluster elements
        p1_friends = np.where(self.cluster_mapping == self.cluster_mapping[p1])[0]
        for elem in p1_friends:
            if elem is not p1:
                for p in range(len(self.points)):
                    self.distances[elem][p] = MAX_DISTANCE
                    self.distances[p][elem] = MAX_DISTANCE
        # Calculate centroids for each element's cluster and recalculate distances based on that
        self.calculate_centroids(self.cluster_mapping)
        self.update_distances(p1, p1_friends)
        return p1, p2

    def update_distances(self, p1, p1_friends):
        for p in range(len(self.points)):
            if p not in p1_friends:
                d = self.calculate_distance(p1, p)
                self.distances[p1][p] = d
                self.distances[p][p1] = d

    def calculate_centroids(self, clusters):
        self.centroids = []
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_mapping == c)[0]
            centroid = self.points[cluster_elems_indexes].mean(axis=0)
            self.centroids.append(centroid)
        return self.centroids

    def calculate_distance(self, c1, c2):
        if self.method==Similarity_Methods.CENTROID:
            return self.euclidean_distance(self.centroids[c1], self.centroids[c2])
        best_d = -1
        total_d = 0
        total_sums = 0

        p1_friends = np.where(self.cluster_mapping == self.cluster_mapping[c1])[0]
        p2_friends = np.where(self.cluster_mapping == self.cluster_mapping[c2])[0]
        for p1 in p1_friends:
            for p2 in p2_friends:
                d = self.euclidean_distance(self.points[p1], self.points[p2])
                total_d += d
                total_sums += 1
                if self.method==Similarity_Methods.MAX:
                    if (d > best_d or best_d < 0):
                        best_d = d
                elif self.method==Similarity_Methods.MIN:
                    if (d < best_d or best_d < 0):
                        best_d = d
        if self.method==Similarity_Methods.AVG:
            return total_d/total_sums
        else:
            return best_d

    def euclidean_distance(self, p1, p2):
        array_1, array_2 = np.array(p1), np.array(p2)
        return np.sqrt(np.sum(np.square(array_1 - array_2)))

    def calculate_cluster_classification(self):
        cluster_classifications = []
        clusters = np.unique(self.cluster_mapping)
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_mapping == c)[0]
            cluster_labels = self.classifications[cluster_elems_indexes]
            p = np.sum(cluster_labels[:,0]) / len(cluster_labels)
            if(p > 0.5):
                cluster_classifications.append(1)
            else:
                cluster_classifications.append(0)
            self.cluster_classifications = cluster_classifications
            self.clusters = clusters.tolist()
            self.centroids = self.calculate_centroids(clusters)

    def predict(self, samples):
        winner_clusters = []
        for s in samples:
            winner_clusters.append(self.get_cluster(s))
        predictions = list(map(lambda c: self.cluster_classifications[self.clusters.index(c)], winner_clusters))
        return predictions

    def get_cluster(self, sample):
        min_distance = -1
        closest_cluster = -1
        for cluster, centroid in zip(self.clusters, self.centroids):
            d = self.euclidean_distance(sample, centroid)
            if min_distance == -1 or d < min_distance:
                min_distance = d
                closest_cluster = cluster
        return closest_cluster
