# Clustering_and_ANOVA
Clustering and ANOVA for 2D data with sample weights using sklearn DBSCAN, Kmeans, and ANOVA.

Sounds were rated by subjects in an experiment. The sounds had their emotional valence and arousal inferred by a random forest model. A clustering analysis was done using the valence and arousal values to plot the data points (1 data point = 1 sound file). The clustering analyses were done using the subject ratings as sample weights. DBSCAN and K-means clustering analysis was done and ANOVA was performed on the clusters to assess the significance of the identified clusters.

The data to be analyzed consists of the following columns in a CSV file. Sound, Valence, Arousal, Average Rating. So there are two dimensions and one sample weight.

The results of the analysis are presented below:

**DBSCAN results:**

When selecting the value for min_pnts, we choose the value of min_pnts that corresponds to the highest silhouette score. The silhouette score is a measure of the relative distance between the clusters. The silhouette score has a value between -1 and 1. A value of -1 means that the clusters are poorly defined, whereas 1 means that they are perfectly distinct from each other. I will find a proper reference for this in the literature. 


k=3 
min_pnts = 7 (see silhouette score vs min_pnts plot below)

Levene's test statistic: 0.15174051434250693
p-value: 0.9792151648425554 (homogeneity confirmed)
ANOVA test statistic: 3.4932787612705143
p-value: 0.005312623308672705 (significant)

 ![Silhoutte](https://github.com/AhmedKhota/Clustering_and_ANOVA/assets/139664971/5e996cb6-02d4-4a33-9066-67038d6597bd)

![clusterafsall](https://github.com/AhmedKhota/Clustering_and_ANOVA/assets/139664971/4910b76c-d4bf-4cfa-8f84-c29bf9eb824a)

![ANOVAALLk3minpnts7](https://github.com/AhmedKhota/Clustering_and_ANOVA/assets/139664971/31f16194-f360-4a9b-9123-80dfd2512ccc)

**Kmeans results:**

k=7 

Levene's test statistic: 0.32584289049488563
p-value: 0.9224618464415357 (homogeneity confirmed)
ANOVA test statistic: 2.2142980209021
p-value: 0.04540809731839591 (significant at p<0.05)

![Uploading clusterafsall.svg…]()

![Uploading ANOVAafsall.svg…]()
