import dohlee.plot as plot; plot.set_style()
import seaborn as sns
from collections import Counter

import itertools
import numpy as np
import pandas as pd

iris = sns.load_dataset('iris')


def set_style_test():
    plot.set_style()


def save_test():
    plot.frequency([1, 2, 2, 3, 3, 3])
    plot.save('tmp.png')


def pca_test():
    data = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    plot.pca(data)
    plot.pca(data, title='Dummy title')
    plot.pca(data, labels=iris.species.values, title='Dummy title')


def boxplot_test():
    ax = plot.get_axis(preset='wide', transpose=True)
    plot.boxplot(data=iris, x='species', y='sepal_length', ax=ax)


def histogram_test():
    ax = plot.get_axis(preset='wide')
    plot.histogram(iris.sepal_length,
                   bins=22,
                   xlabel='Sepal Length',
                   ylabel='Frequency',
                   ax=ax)


def mutation_signature_test():
    data = Counter()
    c_contexts = [p + 'C' + n for (p, n) in itertools.product('ACGT', 'ACGT')]
    t_contexts = [p + 'T' + n for (p, n) in itertools.product('ACGT', 'ACGT')]
    c_alts, t_alts = 'AGT', 'ACG'

    for context, alt in itertools.product(c_contexts, c_alts):
        data[(context, alt)] = np.random.randint(1, 30)
    for context, alt in itertools.product(t_contexts, t_alts):
        data[(context, alt)] = np.random.randint(1, 30)

    ax = plot.get_axis(figsize=(20.4, 3.4))
    plot.mutation_signature(data, ax=ax)
    plot.set_suptitle('Mutational signatures.')


def frequency_test():
    data = [np.random.choice(a=range(10)) for _ in range(100)]
    ax = plot.get_axis(preset='wide')
    plot.frequency(data, ax=ax, xlabel='Your numbers', ylabel='Frequency')


def tsne_test():
    ax = plot.get_axis(preset='wide')
    plot.tsne(
        iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
        ax=ax,
        s=5,
        labels=iris['species']
    )


def linear_regression_test():
    x = np.linspace(0, 1, 100)
    y = 2 * x + 3 + np.random.normal(0, 0.3, len(x))

    ax = plot.get_axis(preset='large')
    plot.linear_regression(x, y, ax=ax)

    ax = plot.get_axis(preset='medium')
    plot.linear_regression(x, y, regression=False, ax=ax)


def stacked_bar_chart_basic_test():
    test_data = pd.DataFrame({
        'Sample': ['S1', 'S2', 'S3'],
        'A': [5, 2, 8],
        'B': [2, 1, 5],
        'C': [1, 4, 2],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'])


def stacked_bar_chart_sort_test():
    test_data = pd.DataFrame({
        'Sample': ['S1', 'S2', 'S3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], sort=True)


def stacked_bar_chart_sort_ascending_test():
    test_data = pd.DataFrame({
        'Sample': ['S1', 'S2', 'S3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], sort=True, reverse=False)


def stacked_bar_chart_single_y_test():
    test_data = pd.DataFrame({
        'Sample': ['S1', 'S2', 'S3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A'])


def stacked_bar_chart_rotate_xticklabels_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], rotate_xticklabels=33)


def stacked_bar_chart_hide_xticklabels_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], xticklabels=False)


def legend_size_small_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], legend_size='small')


def legend_size_medium_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], legend_size='medium')


def legend_size_large_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], legend_size='large')
