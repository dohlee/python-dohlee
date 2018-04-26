from sklearn import datasets

import dohlee.plot as plot

iris = datasets.load_iris()


def pca_test():
    """Test if plot.pca works properly."""
    plot.pca(iris.data)
    plot.pca(iris.data, title='Dummy title')
    plot.pca(iris.data, labels=iris.target, title='Dummy title')
