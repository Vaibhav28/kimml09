import pylab as pl

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA


class SamplePCA:

    def __init__(self):
        return

    def apply(self):
        '''
        Apply PCA decomposition on a matrix
        '''
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        target_names = iris.target_names

        pca = PCA(n_components=2)
        X_r = pca.fit(X).transform(X)

        print 'explained variance ratio (first two components):', \
        pca.explained_variance_ratio_

        pl.figure()
        for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
            pl.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
        pl.legend()
        pl.title('PCA of IRIS dataset')
        pl.savefig('Fig1.png')

if __name__ == "__main__":
    model = SamplePCA()
    model.apply()

'''
Example from
http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#example-decomposition-plot-pca-vs-lda-py

lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print 'explained variance ratio (first two components):', \
    pca.explained_variance_ratio_

pl.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    pl.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
pl.legend()
pl.title('PCA of IRIS dataset')
pl.savefig('Fig1.png')

pl.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    pl.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
pl.legend()
pl.title('LDA of IRIS dataset')

#pl.savefig('Fig1.png')
pl.show()
'''