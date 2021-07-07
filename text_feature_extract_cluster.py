# importing csv module
import csv
import numpy
from scipy import stats

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler



# csv file name
filename = "/home/blin/Downloads/train1en.csv"

# initializing the titles and rows list
fields = []
rows = []

# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        if row[17] == "":
            row[17] = "zero"
            #rows.append(row)
        else:
            rows.append(row)

    # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))

# printing the field names
print('Field names are:' + ', '.join(field for field in fields))


rows_np = numpy.array(rows)
text_to_extract = rows_np[:, 17]
print(text_to_extract)

vec = TfidfVectorizer(stop_words="english")
vec.fit(text_to_extract)
features = vec.transform(text_to_extract)
seed = 0
cls = MiniBatchKMeans(n_clusters=14, random_state=seed)
cls.fit(features)
# predict cluster labels for new dataset
cls.predict(features)

# to get cluster labels for the dataset used while
# training the model (used for models that does not
# support prediction on new dataset).
print(len(cls.labels_))

arr = numpy.array(cls.labels_)
numpy.save("/home/blin/PycharmProjects/nlb/cluster_label.npy", arr)

# reduce the features to 2D
pca = PCA(n_components=2, random_state=seed)
reduced_features = pca.fit_transform(features.toarray())
pred_cls = cls.predict(features)

from sklearn.metrics import silhouette_score
print(silhouette_score(features, labels=pred_cls))

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(cls.cluster_centers_)
plt.scatter(reduced_features[:,0], reduced_features[:,1], c=pred_cls)
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c="b")



plt.show()









