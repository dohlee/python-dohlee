<h1 align="center">python-dohlee</h1>
<p align="center">My personal python library.</p>
<p align="center">
  <a href="https://readthedocs.org/projects/python-dohlee"><img src="https://readthedocs.org/projects/python-dohlee/badge/?style=flat" /></a>
  <a href="https://pypi.python.org/pypi/dohlee"><img src="https://img.shields.io/pypi/v/dohlee.svg" /></a>
  <a href="https://travis-ci.org/dohlee/python-dohlee"><img src="https://travis-ci.org/dohlee/python-dohlee.svg?branch=develop" /></a>
  <a href="https://coveralls.io/r/dohlee/python-dohlee"><img src="https://coveralls.io/repos/dohlee/python-dohlee/badge.svg?branch=develop&service=github" /></a>
</p>

<h2 align="center">Installation</h2>

```
pip install dohlee
```

<h2 align="center">Examples</h2>

### dohlee.plot

Plotting library. Provides simple ways to produce publication-ready plots.

***dohlee.plot.mutation_signature***
```python
import dohlee.plot as plot; plot.set_style()  # Sets plot styles.
ax = plot.get_axis(figsize=(20.4, 3.4))
plot.mutation_signature(data, ax=ax)
```

![mutation_signature](img/mutation_signature.png)

***dohlee.plot.boxplot***
```python
ax = plot.get_axis(preset='wide', transpose=True)
plot.boxplot(data=iris, x='species', y='sepal_length', ax=ax)
```

<p align='center'><img src='img/boxplot.png' style='width:300px'/></p>

***dohlee.plot.histogram***
```python
ax = plot.get_axis(preset='wide')
plot.histogram(iris.sepal_length, bins=22, xlabel='Sepal Length', ylabel='Frequency', ax=ax)
```

<p align='center'><img src='img/histogram.png' /></p>

***dohlee.plot.frequency***
```python
ax = plot.get_axis(preset='wide')
plot.frequency(data, ax=ax, xlabel='Your numbers', ylabel='Frequency')
```

<p align='center'><img src='img/frequency.png'></p>

***dohlee.plot.tsne***
```python
    ax = plot.get_axis(preset='wide')
    plot.tsne(
        iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
        ax=ax,
        s=5,
        labels=iris['species']
    )
```

<p align='center'><img src='img/tsne.png'></p>

<h2 align='center'>Development</h2>

Since this package is updated as needed when I'm doing my research, the development process fits well with TDD cycle.
- When you feel a need to write frequently-used research workflow as a function, write rough tests so that you can be sure that the function you've implemented just meets your need. Write the name of test function as verbose as possible!
-  Run test with following commands. By default, nosetests ignores runnable files while finding test scripts. *--exe* option revokes it.
```shell
nosetests --exe --with-coverage --cover-package=dohlee
```
OR
```shell
tox -e py35,py36
```
- When sufficient progress have been made, test if the package can be published.
```shell
tox
```
- If all tests are passed, distribute the package via PyPI.
```shell
python setup.py sdist
twine upload dist/dohlee-x.x.x.tar.gz
```