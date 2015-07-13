# RSS Article Recommender
A library for suggesting new rss articles to read based on articles in a user's reference manager.
## Getting started

### Dependencies
  * python-feedly (see below)
  * BeautifulSoup
  * pandas
  * nltk (The natural language toolkit)
  * scikit-learn
  * numpy

### Installing python-feedly
The usual method of pip install python-feedly currently fails to install because the package on PyPI is missing the LICENSE.txt. Luckily it can be installed directly from the github repository:
````
pip install git+https//github.com/WarmongeR1/python-feedly.git@master
````
