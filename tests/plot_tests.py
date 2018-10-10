import dohlee.plot as plot; plot.set_style()
import itertools
import tempfile

import numpy as np
import pandas as pd
import seaborn as sns

from collections import Counter
from functools import wraps
from nose.tools import with_setup
from matplotlib.testing.compare import compare_images


def image_comparison(baseline_images, extensions=['png'], tol=1):
    def wrapper(func):
        @wraps(func)
        def decorator(*args, **kwargs):
            results = []
            ax_result = func(*args, **kwargs)
            fp, filename = tempfile.mkstemp(suffix='.png')
            plot.save(filename)

            for baseline_basename in baseline_images:
                for extension in extensions:
                    baseline_image = '.'.join([baseline_basename, extension])

                    result = compare_images(
                        expected=baseline_image,
                        actual=filename,
                        tol=tol,
                    )
        return decorator
    return wrapper


def setup_function():
    pass


def teardown_function():
    plot.clear()


def set_style_test():
    plot.set_style()


@with_setup(setup_function, teardown_function)
def save_test():
    plot.frequency([1, 2, 2, 3, 3, 3])
    plot.save('tmp.png')


@image_comparison(baseline_images=['tests/baseline_images/pca'])
@with_setup(setup_function, teardown_function)
def pca_test():
    iris = sns.load_dataset('iris')
    data = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    plot.pca(data)
    plot.pca(data, title='Dummy title')
    plot.pca(data, labels=iris.species.values, title='Dummy title')


@with_setup(setup_function, teardown_function)
def boxplot_test():
    iris = sns.load_dataset('iris')
    ax = plot.get_axis(preset='wide', transpose=True)
    plot.boxplot(data=iris, x='species', y='sepal_length', ax=ax)


@with_setup(setup_function, teardown_function)
def histogram_test():
    iris = sns.load_dataset('iris')
    ax = plot.get_axis(preset='wide')
    plot.histogram(iris.sepal_length,
                   bins=22,
                   xlabel='Sepal Length',
                   ylabel='Frequency',
                   ax=ax)


@with_setup(setup_function, teardown_function)
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


@with_setup(setup_function, teardown_function)
def frequency_test():
    data = [np.random.choice(a=range(10)) for _ in range(100)]
    ax = plot.get_axis(preset='wide')
    plot.frequency(data, ax=ax, xlabel='Your numbers', ylabel='Frequency')


@with_setup(setup_function, teardown_function)
def tsne_test():
    iris = sns.load_dataset('iris')
    ax = plot.get_axis(preset='wide')
    plot.tsne(
        iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
        ax=ax,
        s=5,
        labels=iris['species']
    )


@with_setup(setup_function, teardown_function)
def linear_regression_test():
    x = np.linspace(0, 1, 100)
    y = 2 * x + 3 + np.random.normal(0, 0.3, len(x))

    ax = plot.get_axis(preset='large')
    plot.linear_regression(x, y, ax=ax)

    ax = plot.get_axis(preset='medium')
    plot.linear_regression(x, y, regression=False, ax=ax)


@with_setup(setup_function, teardown_function)
def stacked_bar_chart_basic_test():
    test_data = pd.DataFrame({
        'Sample': ['S1', 'S2', 'S3'],
        'A': [5, 2, 8],
        'B': [2, 1, 5],
        'C': [1, 4, 2],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'])


@with_setup(setup_function, teardown_function)
def stacked_bar_chart_sort_test():
    test_data = pd.DataFrame({
        'Sample': ['S1', 'S2', 'S3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], sort=True)


@with_setup(setup_function, teardown_function)
def stacked_bar_chart_sort_ascending_test():
    test_data = pd.DataFrame({
        'Sample': ['S1', 'S2', 'S3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], sort=True, reverse=False)


@with_setup(setup_function, teardown_function)
def stacked_bar_chart_single_y_test():
    test_data = pd.DataFrame({
        'Sample': ['S1', 'S2', 'S3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A'])


@with_setup(setup_function, teardown_function)
def stacked_bar_chart_rotate_xticklabels_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], rotate_xticklabels=33)


@with_setup(setup_function, teardown_function)
def stacked_bar_chart_hide_xticklabels_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], xticklabels=False)


@with_setup(setup_function, teardown_function)
def stacked_bar_chart_sort_by_test():
    test_data = pd.DataFrame({
        'Sample': ['S1', 'S2', 'S3', 'S4', 'S5'],
        'A': [1, 3, 5, 1, 5],
        'B': [2, 4, 1, 2, 5],
        'C': [3, 8, 4, 6, 3],
        'Group': ['G1', 'G2', 'G1', 'G1', 'G2'],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], sort=True, sort_by='B')


@with_setup(setup_function, teardown_function)
def stacked_bar_chart_group_test():
    test_data = pd.DataFrame({
        'Sample': ['S1', 'S2', 'S3', 'S4', 'S5'],
        'A': [1, 3, 5, 1, 5],
        'B': [2, 4, 1, 2, 5],
        'C': [3, 8, 4, 6, 3],
        'Group': ['G1', 'G2', 'G1', 'G1', 'G2'],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], group='Group')


@with_setup(setup_function, teardown_function)
def legend_size_small_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], legend_size='small')


@with_setup(setup_function, teardown_function)
def legend_size_medium_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], legend_size='medium')


@with_setup(setup_function, teardown_function)
def legend_size_large_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], legend_size='large')


@with_setup(setup_function, teardown_function)
def legend_title_test():
    test_data = pd.DataFrame({
        'Sample': ['Loooooong label 1', 'Loooooong label 2', 'Loooooong label 3'],
        'A': [1, 3, 5],
        'B': [2, 4, 1],
        'C': [3, 8, 4],
    })
    plot.stacked_bar_chart(data=test_data, x='Sample', y=['A', 'B', 'C'], legend_title='Mutation type')


@with_setup(setup_function, teardown_function)
def umap_test():
    iris = sns.load_dataset('iris')
    ax = plot.get_axis(preset='wide')
    plot.umap(
        iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
        ax=ax,
        s=5,
        labels=iris['species']
    )


@with_setup(setup_function, teardown_function)
def pca_with_no_xyticklabels_test():
    iris = sns.load_dataset('iris')
    ax = plot.get_axis(figsize=(8.4, 8.4))
    plot.pca(
        data=iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
        labels=iris.species,
        s=5,
        ax=ax,
        xticklabels=False,
        yticklabels=False,
    )


@with_setup(setup_function, teardown_function)
def dimensionality_reduction_test():
    iris = sns.load_dataset('iris')
    ax = plot.get_axis(figsize=(8.4 * 3, 8.4))
    plot.dimensionality_reduction(
        data=iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
        labels=iris.species,
        ax=ax,
        xticklabels=False,
        yticklabels=False,
    )


@with_setup(setup_function, teardown_function)
def line_test():
    fmri = sns.load_dataset('fmri')
    ax = plot.get_axis(preset='wide')
    plot.line(
        data=fmri,
        x='timepoint',
        y='signal',
    )
