
class KMeans:
    
    def __init__(self, points, k=2):
        self.points = points 
        self.n = len(points)
        self.k = k
        # Mismo indice que points para indicar a que grupo pertenece cada punto
        self.assignments = self.randomly_assign_group()

    def randomly_assign_group(self)
        return np.random.choice(np.array(range(1, self.k+1)), size=self.n, replace=True)

    def apply_k_means(self):
        was_modified = True

        while (was_modified) {
            groups = self.divide_into_groups()
            was_modified = self.reassign_groups(groups)
        }

    def reassign_groups(self, groups):
        #todo!
        return False

    def divide_into_groups():
        groups = {}
        for i in range(0, len(self.assignments)):
            group_id = self.assignments[i]
            if group_id in groups:
                groups[group_id].points.append(self.points[i])
            else:
                groups[group_id] = {
                    points: [],
                    centroide: []
                }

        self.calculate_centroides(groups)

        return groups

    def calculate_centroids(self, groups):
        for group_id in groups:
            groups[group_id].centroide = self.calculate_centroide(groups[group_id].points)

    def calculate_centroide(self, group_points):
        return np.sum(group_points, axis=0) / len(group_points)

    def calculate_euclidean_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    # Ck - array of points 
    def variation(self, Ck):
        result = 0
        for i in range(0, len(Ck)-1):
            for j in range(i+1, len(Ck)):
                result += variance_bw_points(Ck[i], Ck[j])

        return (1/len(Ck)) * result

    def variance_bw_points(self, p1, p2):
        result = 0

        for i in range(0, len(p1)):
            result += (p1[i]-p2[i])**2
        
        return result
