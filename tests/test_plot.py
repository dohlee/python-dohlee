from sklearn.datasets import load_iris

import dohlee.plot as plot

iris = load_iris()


def test_pca():
    """Test if plot.pca works properly."""
    plot.pca(iris.data)
    plot.pca(iris.data, title='Dummy title')
    plot.pca(iris.data, label=iris.target, title='Dummy title')
