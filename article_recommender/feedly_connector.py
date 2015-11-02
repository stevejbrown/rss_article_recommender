from feedly import client as feedly
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import pandas as pd

class FeedlyStream(object):
    """ FeedlyStream pulls articles from feedly to rank using TrainingData """

    def __init__(self, FEEDLY_CLIENT_TOKEN):
        self.FEEDLY_CLIENT_TOKEN = FEEDLY_CLIENT_TOKEN
        self.client = feedly.FeedlyClient(token = self.FEEDLY_CLIENT_TOKEN, sandbox = False)
        self.streams = {}

    def load_stream(self, feed, num_articles=500):
        ''' Loads unread articles in a feedly stream. Optional parameter num_articles sets how many
            articles to load.
        '''
        stream = self.client.get_feed_content(access_token=self.FEEDLY_CLIENT_TOKEN,
                                            streamId=feed[u'id'], unreadOnly='true',
                                            count=num_articles)
        return stream

    def clean_content(self, content):
        ''' Function to convert raw content from a specific field in feed item to a
            string of words.
            Input: a single string (a raw string of feed content)
            Output: a single string (a processed string of feed content)
        '''

        # Get the longest paragraph tag which usually removes the Journal name, doi, or author list
        soup = BeautifulSoup(content)
        longest_p = max(soup.find_all("p"), key=lambda tag: len(unicode(tag)))

        # Remove HTML
        content_text = longest_p.get_text()

        # Remove non-letters
        letter_only = re.sub("[^a-zA-Z]"," ", content_text)

        # Convert to lower case, split into individual words
        words = letter_only.lower().split()

        # We'll convert this to a set since searching sets is much
        # faster than searching lists
        stops = set(stopwords.words("english"))

        # Remove stop words
        meaningful_words = [w for w in words if not w in stops]

        # Join the words back into one string separated by a space,
        # and return the result
        return(" ".join(meaningful_words))

    def score_stream(self, stream, training_data, vectorizer='count'):
        ''' Scores all the items in a stream and returns a list of [title, url, score] '''
        score_list = []
        for item in stream[u'items']:
            raw_title = item[u'title']
            article_title = self.clean_content(item[u'title'])
            article_url = item[u'alternate'][0][u'href']
            try:
                article_content = self.clean_content(item[u'content'][u'content'])
            # The article may not contain a content field
            except KeyError:
                article_content = ""
            article_words = article_title + " " + article_content
            if vectorizer == 'count':
                score_list.append([raw_title, article_url,
                                   float(training_data.score_article_count(article_words))])
            elif vectorizer == 'tfidf':
                score_list.append([raw_title, article_url,
                                   float(training_data.score_article_tfidf(article_words))])
        return score_list

    def load_category(self, category, articles_per_stream=500):
        ''' Loads all streams in a given category '''
        self.streams[category] = []
        subs = self.client.get_user_subscriptions(self.FEEDLY_CLIENT_TOKEN)
        for feed in subs:
            for cat in feed[u'categories']:
                if cat[u'label'] == category:
                    stream = self.client.get_feed_content(access_token=self.FEEDLY_CLIENT_TOKEN,
                                                        streamId=feed[u'id'],
                                                        unreadOnly='true',
                                                        count=articles_per_stream)
                    self.streams[category].append(stream)
        pass

    def rank_all(self, training_data, vectorizer='count'):
        ''' Ranks all articles in the self.streams dictionary with scoring from training_data.
            Outputs: A sorted list of [title, url, score]
        '''
        rank_list = []
        for category in self.streams:
            for stream in self.streams[category]:
                rank_list = rank_list + self.score_stream(stream, training_data,
                                                          vectorizer=vectorizer)
        rank_df = pd.DataFrame(data=sorted(rank_list, key=lambda entry: entry[-1], reverse=True),
                               columns=['Title', 'Url', 'Score'])
        return rank_df
