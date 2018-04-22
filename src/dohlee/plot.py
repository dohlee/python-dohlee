import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def pca(data, labels=None, title=None, file=None):
    '''Draw a simple principle component analysis plot of the data.'''
    # Fit PCA and get pc's
    pca = PCA(n_components=2)
    pca.fit(data)
    pc = pca.transform(data)

    if title is not None:
        plt.title(title)

    if labels is None:
        plt.scatter(x=pc[:, 0], y=pc[:, 1])

    else:
        # If labels are attached, color them in different colors
        labels = np.array(labels)
        for label in set(labels):
            toDraw = (labels == label)  # only draw these points this time

            plt.scatter(x=pc[toDraw, 0], y=pc[toDraw, 1], label=label)
            plt.legend()

    # show explained variance ratio in the plot axes
    explainedVarianceRatio = pca.explained_variance_ratio_
    plt.xlabel('PC1 ({:.2%})'.format(explainedVarianceRatio[0]))
    plt.ylabel('PC2 ({:.2%})'.format(explainedVarianceRatio[1]))

    if file is None:
        plt.show()
    else:
        plt.savefig(file)
