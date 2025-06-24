import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

palmer = pd.read_csv("data/penguins.csv").dropna()

# i should be able to do PCA by getting a data matrix, centering + scaling, and doing svd

palmer_floats = palmer[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].to_numpy()

scaled_palmer = scale(palmer_floats)

U, s, Vh = svd(scaled_palmer)

scores = U[:, 0:(len(s))] @ np.diag(s) # here, scores @ Vh = scaled_palmer bc lossless bc nrows > ncols = npcs
eigenvals = (s**2) / (len(s) - 1)

# do we get the same thing using sklearn.decomposition.PCA?

pca = PCA()

pca.fit(scaled_palmer)

pca.singular_values_ # this is our s matrix! whoopie
pca.explained_variance_ # not sure how we got this. lol
pca.components_

scores_sklearn = scaled_palmer @ (pca.components_.T) # ok that's also the same. werk

# so can we plot points along first two PCs?

scores_df = pd.DataFrame(scores, columns = ["PC_1", "PC_2", "PC_3", "PC_4"])

# add back scores

palmer_with_pcs = pd.merge(palmer, scores_df, left_index = True, right_index = True)

plt.figure()
sns.scatterplot(data = palmer_with_pcs, x = 'PC_1', y = 'PC_2', hue = 'species', style = 'sex')


## TOY DATA ----
# we can also do the same process on our toy data

toys = np.array([[1, -1], [-1, 1], [2, -2], [-2, 2]])

U, s, Vh = svd(toys)

scores = U[:, 0:(len(s))] @ np.diag(s) # here, scores @ Vh = scaled_palmer bc lossless bc nrows > ncols = npcs
eigenvals = (s**2) / (len(s) - 1)

# here the eigenvales are (basically) the same as what you get on paper
# so is my U1 vector
# my signs are flipped for V matrix? but ig that doesn't matter (?)

pca = PCA()

pca.fit(toys)

pca.singular_values_ # this is our s matrix! whoopie
pca.explained_variance_ # not sure how we got this. lol
pca.components_
