import numpy as np

class KMeans:

    def __init__(self, points, classifications, k=2):
        self.points = points
        self.classifications = classifications
        self.n = len(points)
        self.k = k
        # Mismo indice que points para indicar a que grupo pertenece cada punto
        self.assignments = self.randomly_assign_group()

    def randomly_assign_group(self):
        # Ensure all K are in selected at least once
        range_arr = np.array(range(0, self.k))
        selection = np.copy(range_arr)
        return np.append(selection, np.random.choice((range_arr), size=(self.n-self.k), replace=True))

    def apply(self):
        was_modified = True
        # While there are still group changes, continue the algorithm
        while was_modified:
            groups, centroides = self.divide_into_groups()
            was_modified = self.reassign_groups(groups, centroides)
        # Measure positivity rate of each cluster
        positivity = {}
        for group_index in groups:
            positivity[group_index] = 0
            for point_index in groups[group_index]:
                positivity[group_index] += self.classifications[point_index]
            positivity[group_index] = positivity[group_index]/len(groups[group_index])
            if (positivity[group_index] > 0.5):
                positivity[group_index] = 1
            else:
                positivity[group_index] = 0
        self.positivity = positivity
        self.centroids = centroides

        # Return array of assignments
        self.classes = []
        for cluster in np.unique(self.assignments):
            self.classes.append([])
            for i in range(len(self.assignments)):
                if self.assignments[i]==cluster:
                    self.classes[-1].append(i)
        return self.classes

    def reassign_groups(self, groups, centroides):
        # Create a new assignment array where the new group ids will be stored
        new_assigments = np.copy(self.assignments)

        for group_id in groups:
            for point_idx in groups[group_id]:
                # Point we are analyzing
                point = self.points[point_idx]
                # For each point, check which centroide is nearest
                min_dist = None
                for centroide_id in centroides:
                    # Distance of the point with the centroide in question
                    dist = self.calculate_euclidean_distance(centroides[centroide_id], point)
                    # Saving the group of the centroide that has the minimum distance
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        new_assigments[point_idx] = centroide_id
        # If there were no changes to the group, no modifications occurred
        was_modified = not np.array_equal(new_assigments, self.assignments)
        self.assignments = new_assigments
        return was_modified

    def divide_into_groups(self):
        groups = {}
        for i in range(0, len(self.assignments)):
            group_id = self.assignments[i]
            if group_id in groups:
                groups[group_id].append(i)
            else:
                groups[group_id] = [i]
        centroides = self.calculate_centroides(groups)
        return groups, centroides

    def calculate_centroides(self, groups):
        centroides = {}
        for group_id in groups:
            centroides[group_id] = self.calculate_centroide(groups[group_id])
        return centroides

    def calculate_centroide(self, group_points_idx):
        group_points = list(map(lambda i: self.points[i], group_points_idx))
        return np.sum(group_points, axis=0) / len(group_points)

    def calculate_euclidean_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    # Ck - array of points
    def variation(self, Ck):
        result = 0
        for i in range(0, len(Ck)-1):
            for j in range(i+1, len(Ck)):
                result += self.variance_bw_points(Ck[i], Ck[j])

        return (1/len(Ck)) * result

    def variance_bw_points(self, p1, p2):
        result = 0

        for i in range(0, len(p1)):
            result += (p1[i]-p2[i])**2

        return result

    def predict(self, samples):
        predictions = []
        for s in samples:
            min_dist = None
            closest_cluster = None
            for centroid_id in self.centroids:
                # Distance of the point with the centroide in question
                dist = self.calculate_euclidean_distance(self.centroids[centroid_id], s)
                # Saving the group of the centroide that has the minimum distance
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    closest_cluster = centroid_id
            predictions.append(self.positivity[closest_cluster])
        return predictions
