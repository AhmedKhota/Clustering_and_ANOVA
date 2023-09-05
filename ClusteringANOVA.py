import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

afs = "all"

class DBSCANalyser:
    
    def __init__(self, data: pd.DataFrame):
        self.data = data[['V','A']].values
        self.sample_weights = data['avrating'].values
        self.k = 0
        self.min_pnts = 0
        self.eps = 0
        self.clusters = []
        self.clusters_ratings = []

    def data_check(self):
        print("checking sample weights:")
        print(type(self.sample_weights))
        print(self.sample_weights.shape)
        print("Done checking sample weights")
        print("checking V,A:")
        print(type(self.data))
        print(self.data.shape)
        print("Done checking V,A")

    def get_k(self):
        self.k = int(input("enter the value of k for KNN"))

    def set_eps(self):
        neigh = NearestNeighbors(n_neighbors=self.k)
        nbrs = neigh.fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        distance_desc = sorted(distances[:, self.k-1], reverse = True)
        kneedle = KneeLocator(range(1, len(distance_desc)+1), distance_desc, S=1.0, curve = "convex", direction = "decreasing")
        #kneedle.plot_knee_normalized()
        print("eps is: ", kneedle.knee_y)
        self.eps = kneedle.knee_y

    def set_min_pnts(self):
        # Define the range of min_samples values to test
        min_samples_range = range(2, 11)
        silhouette_scores = []

        # Loop through different min_samples values
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=self.eps, min_samples=min_samples)
            labels = dbscan.fit_predict(self.data, sample_weight=self.sample_weights)
            silhouette_avg = silhouette_score(self.data, labels)
            silhouette_scores.append(silhouette_avg)
        # Plot the results
        plt.plot(min_samples_range, silhouette_scores, marker='o')
        plt.xlabel('min_samples')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs min_samples')
        plt.show()
        self.min_pnts = int(input("enter the min_pnts value from the plot"))

    def plot_clusters(self):
        m = DBSCAN(eps=self.eps, min_samples=self.min_pnts)
        m.fit(self.data,sample_weight = self.sample_weights)
        colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'lightgrey'] #afs1
        vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
        clusters = m.labels_
        self.clusters = clusters
        markers = ['o', '^', 's', 'x', 'p', 'P', '*', '+', 'X', '8'] 
        col = vectorizer(clusters)
        for i, c in enumerate(np.unique(col)):
            plt.scatter(self.data[:,0][col==c],self.data[:,1][col==c],c=col[col==c], marker=markers[i])
        plt.xlabel('Valence', fontsize=15)
        plt.ylabel('Arousal', fontsize=15)
        plt.title('DBSCAN result of Afs{0}'.format(afs), fontsize=15) 
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xticks(np.arange(0, 1, step=0.1), fontsize=15)
        plt.yticks(np.arange(0, 1, step=0.1), fontsize=15)
        plt.savefig('New DBSCAN/clusterafs{0}.svg'.format(afs))
        plt.show()

    def check_clusters(self):
        self.clusters_ratings = [[x, y] for x, y in zip(self.sample_weights, self.clusters)]
        print(np.array(self.clusters_ratings).shape)
        print(type(self.clusters_ratings))
        print(self.clusters_ratings[0])

    def cluster_results(self):
        return self.clusters_ratings

class KMEANSanalyser:
    
    def __init__(self, data: pd.DataFrame):
        self.data = data[['V', 'A']].values
        self.sample_weights = data['avrating'].values
        self.k = 0
        self.clusters = []
        self.clusters_ratings = []

    def get_k(self):
        self.k = int(input("Enter the number of clusters (k) for K-Means"))

    def fit_clusters(self):
        kmeans = KMeans(n_clusters=self.k)
        labels = kmeans.fit_predict(self.data, sample_weight=self.sample_weights)
        self.clusters = labels

    def plot_clusters(self):
        m = KMeans(n_clusters=self.k, n_init=10, random_state=42)
        labels = m.fit_predict(self.data, sample_weight=self.sample_weights)#change code for KMEANS
        colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'lightgrey'] #afs1
        vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
        clusters = labels#change code for KMEANS
        self.clusters = clusters
        markers = ['o', '^', 's', 'x', 'p', 'P', '*', '+', 'X', '8'] #afs1
        col = vectorizer(clusters)
        for i, c in enumerate(np.unique(col)):
            plt.scatter(self.data[:,0][col==c],self.data[:,1][col==c],c=col[col==c], marker=markers[i])
        plt.xlabel('Valence', fontsize=15)
        plt.ylabel('Arousal', fontsize=15)
        plt.title('KMEANS result of Afs{0}'.format(afs), fontsize=15) #afs1 #change to dynamically update scenario number
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xticks(np.arange(0, 1, step=0.1), fontsize=15)
        plt.yticks(np.arange(0, 1, step=0.1), fontsize=15)
        plt.savefig('New DBSCAN/Kmeans/clusterafs{0}.svg'.format(afs))#change to dynamically update scenario number
        plt.show()

    def check_clusters(self):
        self.clusters_ratings = [[x, y] for x, y in zip(self.sample_weights, self.clusters)]
        print(np.array(self.clusters_ratings).shape)
        print(type(self.clusters_ratings))
        print(self.clusters_ratings[0])

    def cluster_results(self):
        return self.clusters_ratings

class ANOVA:

    def __init__(self, clusters: list):
        self.clusters = clusters
        self.create_clusters()

    def create_clusters(self):
        # Create an empty dictionary to store clusters as keys and lists of values as values
        cluster_lists = {}
        # Iterate through the data and populate the dictionary
        for value, cluster in self.clusters:
            if cluster in cluster_lists:
                cluster_lists[cluster].append(value)
            else:
                cluster_lists[cluster] = [value]
        # Convert the dictionary values to lists
        cluster_lists = list(cluster_lists.values())
        return cluster_lists

    def levene(self):
        levene_result = stats.levene(*self.create_clusters())
        print("Levene's test statistic:", levene_result.statistic)
        print("p-value:", levene_result.pvalue)

    def onewayanova(self):
        anova_result = stats.f_oneway(*self.create_clusters())
         # Print the result
        print("ANOVA test statistic:", anova_result.statistic)
        print("p-value:", anova_result.pvalue)

    def plotresults(self):
        fsize = 18
        fig = plt.figure(figsize= (10, 10))
        ax = fig.add_subplot(111)
        ax.set_title("scenario {0}".format(afs), fontsize=9) #afs2 #change to dynamically update scenario number
        ax.set
        data = [*self.create_clusters()]
        num_clusters = len(data)
        cluster_labels = [f"cluster {i}" for i in range(num_clusters)]
        ax.boxplot(data,
                showmeans= True)
        ax.set_xticklabels(labels= cluster_labels, fontsize=8)
        #ax.set_yicklabels(labels= 'average rating', font = font, fontsize=9)
        #plt.xlabel("DBSCAN cluster",font = font, fontsize=9)
        plt.ylabel("average rating", fontsize=fsize)
        #plt.savefig('New DBSCAN/ANOVAafs{0}.svg'.format(afs)) #change to dynamically update scenario number #DBSCAN
        plt.savefig('New DBSCAN/Kmeans/ANOVAafs{0}.svg'.format(afs)) #change to dynamically update scenario number #Kmeans
        plt.show()

if __name__ == '__main__':
    #afs = afs
    data = pd.read_csv('AudioSetExpSounds/afs{0}labels.csv'.format(afs))
    #dbscanalyser = DBSCANalyser(data)
    #dbscanalyser.data_check()
    #dbscanalyser.get_k()
    #dbscanalyser.set_eps()
    #dbscanalyser.set_min_pnts()
    #dbscanalyser.plot_clusters()
    #dbscanalyser.check_clusters()
    #cluster_data = dbscanalyser.cluster_results()

    kmeansanalyser = KMEANSanalyser(data)
    kmeansanalyser.get_k()
    kmeansanalyser.fit_clusters()
    kmeansanalyser.plot_clusters()
    kmeansanalyser.check_clusters()
    cluster_data_kmeans = kmeansanalyser.cluster_results()
    

    #anova = ANOVA(cluster_data)
    #anova.levene()
    #anova.onewayanova()
    #anova.plotresults()

    anova = ANOVA(cluster_data_kmeans)
    anova.levene()
    anova.onewayanova()
    anova.plotresults()






