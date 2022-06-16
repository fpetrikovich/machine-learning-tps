import numpy as np
import itertools
from config.constants import Similarity_Methods

class Hierarchy:
    def __init__(self, points, classifications, method=Similarity_Methods.CENTROID):
        self.clusters = []
        for i in range(len(points)):
            p = points[i]
            c = classifications[i]
            self.clusters.append({'content': [p], 'positives': c, 'depth': None, 'left': None, 'right': None, 'centroid': p})
        self.n = len(self.clusters)
        self.method = method

    def apply(self):
        while (self.n > 1):
            print("n =", self.n)
            min_d = -1
            min_c1 = None
            min_c2 = None
            for c1, c2 in itertools.combinations(self.clusters, 2):
                d = self.calculate_distance(c1, c2)
                if (d < min_d or min_d < 0):
                    min_d = d
                    min_c1 = c1
                    min_c2 = c2
            self.unite(min_c1, min_c2, min_d)
        #self.print_cluster(self.clusters[0], 0)
        return self.clusters[0]

    def print_cluster(self, c, level):
        for i in range(level):
            print("-", end="")
        if c['left'] is None:
            print(c['content'])
        else:
            print("x")
            self.print_cluster(c['left'], level+1)
            self.print_cluster(c['right'], level+1)

    def unite(self, c1, c2, d):
        c1 = c1.copy()
        c2 = c2.copy()
        elements = c1['content'] + c2['content']
        positives = c1['positives'] + c2['positives']
        fork = {'depth': d, 'left': c1, 'right': c2, 'content': elements, 'positives': positives, 'centroid': self.calculate_centroid(elements)}
        i = 0
        while (i < self.n):
            if self.clusters[i]['content'] is c1['content'] or self.clusters[i]['content'] is c2['content']:
                self.clusters.pop(i)
                self.n -= 1
            else:
                i += 1
        self.clusters.append(fork)
        self.n = len(self.clusters)

    def calculate_centroid(self, elements):
        return np.sum(elements, axis=0) / len(elements)

    def euclidean_distance(self, p1, p2):
        array_1, array_2 = np.array(p1), np.array(p2)
        return np.sqrt(np.sum(np.square(array_1 - array_2)))

    def calculate_distance(self, c1, c2):
        if self.method==Similarity_Methods.CENTROID:
            return self.euclidean_distance(c1['centroid'], c2['centroid'])
        best_d = -1
        total_d = 0
        total_sums = 0
        for p1 in c1['content']:
            for p2 in c2['content']:
                d = self.euclidean_distance(p1, p2)
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

    def test(self, input):
        min_d = -1
        min_c = None
        clusters = [self.clusters[0]['left'], self.clusters[0]['right']]
        positivity_L = clusters[0]['positives']/len(clusters[0]['content'])
        positivity_R = clusters[1]['positives']/len(clusters[1]['content'])
        if positivity_L < 0.5: positivity_L = 0
        else: positivity_L = 1
        if positivity_R < 0.5: positivity_R = 0
        else: positivity_R = 1
        positivity = [positivity_L, positivity_R]

        for i in range(len(clusters)):
            d = self.calculate_distance({'content': [input], 'centroid': input}, clusters[i])
            if (d < min_d or min_d < 0):
                min_d = d
                min_c = positivity[i]
        return i
