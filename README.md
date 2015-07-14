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

### Inputs

To use the included Example ipython notebook, store your feedly developer token and exported zotero library to a folder called 'inputs' inside the project folder ('rss_article_recommender').

### Get a feedly developer token
To easily interact with feedly you should generate a developer token from https://feedly.com/v3/auth/dev. Using the developer token we can skip authenticating through OAuth 2.0 and instead make requests to feedly directly (much easier). The developer token limits us to working with just our own account and is for personal use, which is fine for our purposes.

Save your developer to the first line of a file called 'feedly_client_token.txt'. Stores this in the inputs folder.

### Exporting Zotero library
To export your Zotero library click on the gear in the Zotero toolbar, select Export library, then choose the CSV format. If you are using the Zotero standalone client right-click My Library, choose Export Library, and then the CSV format.

Save the exported Zotero library as 'zotero_library.csv' and store it in the inputs folder.

### Run the example notebook
You should now be able to run the example notebook and start generating your very own recommendations!
