import sys
import math
import os
import heapq
import itertools
import csv
import random
from math import sqrt
from math import fsum

class Clustering:
    def __init__(self, ipt_data, ipt_k):
        self.input_file_name = ipt_data
        self.k = ipt_k
        self.dataset = None
        self.dataset_size = 0
        self.dimension = 0
        self.heap = []
        self.clusters = []
        self.gold_standard = {}

    def initialize_heap(self):
        self.heap = []

    def initialize(self):
        """
        Initialize and check parameters
        """
        # check file exist and if it's a file or dir
        if not os.path.isfile(self.input_file_name):
            self.quit("Input file doesn't exist or it's not a file")

        self.dataset, self.clusters, self.gold_standard = self.load_data(self.input_file_name)
        self.dataset_size = len(self.dataset)

        if self.dataset_size == 0:
            self.quit("Input file doesn't include any data")

        if self.k == 0:
            self.quit("k = 0, no cluster will be generated")

        if self.k > self.dataset_size:
            self.quit("k is larger than the number of existing clusters")

        self.dimension = len(self.dataset[0]["data"])

        if self.dimension == 0:
            self.quit("dimension for dataset cannot be zero")

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                      Hierarchical Clustering Functions                       """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def euclidean_distance(self, data_point_one, data_point_two):
        """
        euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
        assume that two data points have same dimension
        """
        size = len(data_point_one)
        result = 0.0
        for i in range(size):
            f1 = float(data_point_one[i])   # feature for data one
            f2 = float(data_point_two[i])   # feature for data two
            tmp = f1 - f2
            result += pow(tmp, 2)
        result = math.sqrt(result)
        return result

    def compute_pairwise_distance(self, dataset):
        result = []
        dataset_size = len(dataset)
        for i in range(dataset_size-1):    # ignore last i
            for j in range(i+1, dataset_size):     # ignore duplication
                dist = self.euclidean_distance(dataset[i]["data"], dataset[j]["data"])

                # duplicate dist, need to be remove, and there is no difference to use tuple only
                # leave second dist here is to take up a position for tie selection
                result.append( (dist, [dist, [[i], [j]]]) )

        return result
                
    def build_priority_queue(self, distance_list):
        heapq.heapify(distance_list)
        self.heap = distance_list
        return self.heap


    def compute_centroid(self, dataset, data_points_index):
        size = len(data_points_index)
        dim = self.dimension
        centroid = [0.0]*dim
        for idx in data_points_index:
            dim_data = dataset[idx]["data"]
            for i in range(dim):
                centroid[i] += float(dim_data[i])
        for i in range(dim):
            centroid[i] /= size
        return centroid

    def Single_Linkage_clustering(self):
        """
        Main Process for Single_Linkage_Clustering
        """
        
        dataset = self.dataset
        current_clusters = self.clusters
        #print(current_clusters)
        old_clusters = []
        heap = hc.compute_pairwise_distance(dataset)
        heap = hc.build_priority_queue(heap)
        dummy_clusters = {}
        while len(current_clusters) > self.k:
            dist, min_item = heapq.heappop(heap)
            # pair_dist = min_item[0]
            pair_data = min_item[1]
            #dummy_clusters = {}
            #print(pair_data)
            # judge if include old cluster
            if not self.valid_heap_node(min_item, old_clusters):
                continue

            new_cluster = {}
            new_cluster_elements = sum(pair_data, [])
            new_cluster_cendroid = self.compute_centroid(dataset, new_cluster_elements)
            
            new_cluster_elements.sort()
            new_cluster.setdefault("centroid", new_cluster_cendroid)
            new_cluster.setdefault("elements", new_cluster_elements)
            
            for pair_item in pair_data:
                old_clusters.append(pair_item)
                dummy_clusters[str(pair_item)] = current_clusters[str(pair_item)]       #---------------------------------------------------------------------
                del current_clusters[str(pair_item)]
            
            self.single_add_heap_entry(heap, new_cluster, current_clusters, dummy_clusters)
            current_clusters[str(new_cluster_elements)] = new_cluster
        #current_clusters.sort()
        #print(current_clusters)
        return current_clusters

    def Complete_Linkage_clustering(self):
        """
        Main Process for Complete_Linkage_Clustering
        """
        dataset = self.dataset
        current_clusters = self.clusters
        #print(current_clusters)
        old_clusters = []
        heap = hc.compute_pairwise_distance(dataset)
        heap = hc.build_priority_queue(heap)
        dummy_clusters = {}
        while len(current_clusters) > self.k:
            dist, min_item = heapq.heappop(heap)
            # pair_dist = min_item[0]
            pair_data = min_item[1]
            #dummy_clusters = {}
            #print(pair_data)
            # judge if include old cluster
            if not self.valid_heap_node(min_item, old_clusters):
                continue

            new_cluster = {}
            new_cluster_elements = sum(pair_data, [])
            new_cluster_cendroid = self.compute_centroid(dataset, new_cluster_elements)
            
            new_cluster_elements.sort()
            new_cluster.setdefault("centroid", new_cluster_cendroid)
            new_cluster.setdefault("elements", new_cluster_elements)
            
            for pair_item in pair_data:
                old_clusters.append(pair_item)
                dummy_clusters[str(pair_item)] = current_clusters[str(pair_item)]       #---------------------------------------------------------------------
                del current_clusters[str(pair_item)]
            
            self.complete_add_heap_entry(heap, new_cluster, current_clusters, dummy_clusters)
            current_clusters[str(new_cluster_elements)] = new_cluster
        #current_clusters.sort()
        return current_clusters

    def Average_Linkage_clustering(self):
        """
        Main Process for Average_Linkage_Clustering
        """
        
        dataset = self.dataset
        current_clusters = self.clusters
        #print(current_clusters)
        old_clusters = []
        heap = hc.compute_pairwise_distance(dataset)
        heap = hc.build_priority_queue(heap)
        dummy_clusters = {}
        while len(current_clusters) > self.k:
            dist, min_item = heapq.heappop(heap)
            # pair_dist = min_item[0]
            pair_data = min_item[1]
            #dummy_clusters = {}
            #print(pair_data)
            # judge if include old cluster
            if not self.valid_heap_node(min_item, old_clusters):
                continue

            new_cluster = {}
            new_cluster_elements = sum(pair_data, [])
            new_cluster_cendroid = self.compute_centroid(dataset, new_cluster_elements)
            
            new_cluster_elements.sort()
            new_cluster.setdefault("centroid", new_cluster_cendroid)
            new_cluster.setdefault("elements", new_cluster_elements)
            
            for pair_item in pair_data:
                old_clusters.append(pair_item)
                dummy_clusters[str(pair_item)] = current_clusters[str(pair_item)]       #---------------------------------------------------------------------
                del current_clusters[str(pair_item)]
            
            self.average_add_heap_entry(heap, new_cluster, current_clusters, dummy_clusters)
            current_clusters[str(new_cluster_elements)] = new_cluster
        #current_clusters.sort()
        return current_clusters
            
    def valid_heap_node(self, heap_node, old_clusters):
        pair_dist = heap_node[0]
        pair_data = heap_node[1]
        for old_cluster in old_clusters:
            if old_cluster in pair_data:
                return False
        return True

    def single_add_heap_entry(self, heap, new_cluster, current_clusters, dummy_clusters):

        for ex_cluster in current_clusters.values():
            dist = 10000000
            new_heap_entry = []
            for elem in new_cluster["elements"]:
                li = [elem]
                temp = self.euclidean_distance(ex_cluster["centroid"], dummy_clusters[str(li)]["centroid"])
                if(temp < dist):
                    dist = temp
            new_heap_entry.append(dist)
            new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
            heapq.heappush(heap, (dist, new_heap_entry))

    def complete_add_heap_entry(self, heap, new_cluster, current_clusters, dummy_clusters):

        for ex_cluster in current_clusters.values():
            dist = 0
            new_heap_entry = []
            
            for elem in new_cluster["elements"]:
                li = [elem]
                temp = self.euclidean_distance(ex_cluster["centroid"], dummy_clusters[str(li)]["centroid"])
                if(temp > dist):
                    dist = temp
            new_heap_entry.append(dist)
            new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
            heapq.heappush(heap, (dist, new_heap_entry))
            
    def average_add_heap_entry(self, heap, new_cluster, current_clusters, dummy_clusters):

        for ex_cluster in current_clusters.values():
            dist = 0
            new_heap_entry = []
            count =0
            
            for elem in new_cluster["elements"]:
                li = [elem]
                temp = self.euclidean_distance(ex_cluster["centroid"], dummy_clusters[str(li)]["centroid"])
                dist +=temp
                count+=1
            dist = dist/count
            new_heap_entry.append(dist)
            new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
            heapq.heappush(heap, (dist, new_heap_entry))
                
            
            

    def Hamming_distance(self, current_clusters):
        gold_standard = self.gold_standard
        current_clustes_pairs = []

        for (current_cluster_key, current_cluster_value) in current_clusters.items():
            tmp = list(itertools.combinations(current_cluster_value["elements"], 2))
            current_clustes_pairs.extend(tmp)
        tp_fp = len(current_clustes_pairs)

        gold_standard_pairs = []
        for (gold_standard_key, gold_standard_value) in gold_standard.items():
            tmp = list(itertools.combinations(gold_standard_value, 2))
            gold_standard_pairs.extend(tmp)
        tp_fn = len(gold_standard_pairs)


        total = math.factorial(len(self.dataset))/ (math.factorial(len(self.dataset)-2) * math.factorial(2))
        #print(total)
        tp = 0.0
        for ccp in current_clustes_pairs:
            if ccp not in gold_standard_pairs:
                tp += 1
        #print(tp)
        hamming_distance = (tp)/total

        return hamming_distance

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                             Helper Functions                                 """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def load_data(self, input_file_name):
        """
        load data and do some preparations
        """
        with open(input_file_name) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            dataset = []
            clusters = {}
            gold_standard = {}
            id = 0
            for row in readCSV:
                iris_class = row[-1]

                '''data = {}
                data.setdefault("id", id)   # duplicate
                data.setdefault("data", row[:-1])
                data.setdefault("class", row[-1])
                dataset.append(data)'''

                data = {}
                data.setdefault("id", id)   # duplicate
                d = []
                for i in range(len(row)-1):
                    d.append(row[i])

                data.setdefault("data", d)
                #print(d)
                data.setdefault("class", row[-1])
                dataset.append(data)

                clusters_key = str([id])
                clusters.setdefault(clusters_key, {})
                clusters[clusters_key].setdefault("centroid", row[:-1])
                clusters[clusters_key].setdefault("elements", [id])

                gold_standard.setdefault(iris_class, [])
                gold_standard[iris_class].append(id)

                id += 1
            #print(len(gold_standard))
        return dataset, clusters, gold_standard

    def display(self, current_clusters, hamming_distance):
        print("Hamming Distance :" ,hamming_distance)
        #print(recall)
        clusters = current_clusters.values()
        for cluster in clusters:
            cluster["elements"].sort()
            print(cluster["elements"])



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                               Main Method                                    """
"""                                                                              """    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':

    #hc = Clustering("Iris.csv", 3)
    hc = Clustering('drinking_fountains.csv', 2)
    hc.initialize()
    print("---------------------------------------------------------------------------------------------------------")
    print("Single Linkage Clustering ")
    new_clusters = hc.Single_Linkage_clustering()
    hamming_distance = hc.Hamming_distance(new_clusters)
    hc.display(new_clusters, hamming_distance)
    print("---------------------------------------------------------------------------------------------------------")
    hc.initialize()
    print("Compelte Linkage Clustering ")
    current_clusters1 = hc.Complete_Linkage_clustering()
    hamming_distance1 = hc.Hamming_distance(current_clusters1)
    hc.display(current_clusters1, hamming_distance1)
    print("---------------------------------------------------------------------------------------------------------")
    hc.initialize()
    print("Average Linkage Clustering ")
    current_clusters2 = hc.Average_Linkage_clustering()
    hamming_distance2 = hc.Hamming_distance(current_clusters2)
    hc.display(current_clusters2, hamming_distance2)


gold_standard={}
inp = {}
k = 2

#with open('Iris.csv') as csvfile:
with open('drinking_fountains.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    j = 1
    for row in readCSV:
        dict = {}
        iris_class = row[-1]
        for i in range(0, len(row)-1):
            dict[i] = float(row[i]);
        inp[j] = dict
        j += 1
        gold_standard.setdefault(iris_class, [])
        gold_standard[iris_class].append(j)
    '''print(inp)
    print("------------------------------------")
    print(inp.values())'''

def initialize_kmeansplusplus(inputs, k):
    centroids = random.sample(inputs, 1)
    while(len(centroids) < k):
        D2 = []
        for x in inputs:
            distance = 100000000000
            for y in centroids:
                curr_dist = calculate_euclidean_distance(x, y)
                if(curr_dist < distance):
                    distance = curr_dist
            D2.append(distance**2)
        probs = []
        s = fsum(D2)
        for d in D2:
            probs.append(d/s)
        cum_sum = []
        cum_sum.append(probs[0])
        for i in range(1, len(probs)):
            cum_sum.append(cum_sum[i-1]+probs[i])
        rand = random.random()
        index = -1
        for i in range(0, len(cum_sum)):
            if(cum_sum[i] >= rand):
                index = i
                break
        centroids.append(inputs[index])
    return centroids


def Hamming_distance(current_clusters, gold_standard):
        #gold_standard = self.gold_standard
        current_clustes_pairs = []

        for (current_cluster_key, current_cluster_value) in current_clusters.items():
            tmp = list(itertools.combinations(current_cluster_value, 2))
            current_clustes_pairs.extend(tmp)
        tp_fp = len(current_clustes_pairs)

        gold_standard_pairs = []
        for (gold_standard_key, gold_standard_value) in gold_standard.items():
            tmp = list(itertools.combinations(gold_standard_value, 2))
            gold_standard_pairs.extend(tmp)
        tp_fn = len(gold_standard_pairs)


        total = math.factorial(len(inp))/ (math.factorial(len(inp)-2) * math.factorial(2))
        #print(total)
        tp = 0.0
        for ccp in current_clustes_pairs:
            if ccp not in gold_standard_pairs:
                tp += 1
        #print(tp)
        hamming_distance = (tp)/total
        print("------------------------------------------------")
        print("Hamming Distance :",hamming_distance)
        print("------------------------------------------------")
        #return hamming_distance


def has_converged(new_centroids, old_centroids):
    return (set([tuple(a) for a in new_centroids]) == set([tuple(a) for a in old_centroids]))

def calculate_euclidean_distance(x, centroid):
    sqdist = 0.0
    for i, v in x.items():
        sqdist += (v-centroid[i]) ** 2
    return sqrt(sqdist)

def initialize_centroids(inputs, k):
    centroids = random.sample(inputs, k)
    return centroids

def distribute_points(inputs, centroids):
    clusters = {}
    for x in inputs:
        min = 10000000000
        best_centroid = -1
        for i in range(0, len(centroids)):
            dist = calculate_euclidean_distance(x, centroids[i])
            if(dist < min):
                min = dist
                best_centroid = i
        if best_centroid in clusters:
            clusters[best_centroid].append(x)
        else:
            clusters[best_centroid] = [x]
    return clusters

def reevaluate_centers(centroids, clusters):
    new_centroids = []
    for k in clusters.keys():
        num_rows = len(clusters[k][0])
        new_centroids.append(mean(clusters[k], num_rows))
    return new_centroids

def mean(cluster, l):
    centroid = [0.] * l
    n = 0
    for x in cluster:
        for i, v in x.items():
            centroid[i] += v
        n += 1
    for i in range(0, l):
        centroid[i] /= n
    return centroid

def form_clusters(inputs, centroids):
    clusters = {}
    for y, x in inputs.items():
        min = 10000000000
        best_centroid = -1
        for i in range(0, len(centroids)):
            dist = calculate_euclidean_distance(x, centroids[i])
            if(dist < min):
                min = dist
                best_centroid = i
        if best_centroid in clusters:
            clusters[best_centroid].append(y)
        else:
            clusters[best_centroid] = [y]
    Hamming_distance(clusters,gold_standard)
    return clusters

print("-----------------------------------------------------------")
print(" ------------------K- means Plus Plus----------------------")
print("-----------------------------------------------------------")
def find_clusters_k_plusplus(inpu, k):
    inputs = list(inp.values())
    old_centroids = []
    clusters = {}
    centroids = initialize_kmeansplusplus(inputs, k)
    print("-----------------------------------------------------------")
    print("Initial Centroids : ", centroids)
    while not has_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = distribute_points(inputs, centroids)
        centroids = reevaluate_centers(old_centroids, clusters)
        print("centroid is shifting :")
        print(centroids)
        print("-------------------------------------------")
    print("-----------------------------------------------------------")
    print(" Final Centroids :",centroids)
    print("-----------------------------------------------------------")
    return form_clusters(inp, centroids)
#print(initialize_centroids(inputs, k))
print(find_clusters_k_plusplus(inp, k))


print("-----------------------------------------------------------")
print(" --------------------Lloyd's Algorithm---------------------")
print("-----------------------------------------------------------")
def find_clusters_lloyds(inpu, k):
    inputs = list(inp.values())
    old_centroids = []
    clusters = {}
    centroids = initialize_centroids(inputs, k)
    print("-----------------------------------------------------------")
    print("Initial Centroids :", centroids)
    while not has_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = distribute_points(inputs, centroids)
        centroids = reevaluate_centers(old_centroids, clusters)
        print("centroid is shifting :")
        print(centroids)
        print("-------------------------------------------")
    print("-----------------------------------------------------------")
    print("Final Centroids :",centroids)
    print("-----------------------------------------------------------")
    return form_clusters(inp, centroids)
#print(initialize_centroids(inputs, k))
print(find_clusters_lloyds(inp, k))
    
