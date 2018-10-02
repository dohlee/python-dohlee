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
