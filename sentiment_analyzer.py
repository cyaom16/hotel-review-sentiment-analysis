from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import PerceptronTagger
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import RegexpParser
from collections import Counter
from tqdm import tqdm

import sklearn.metrics as metrics
import pandas as pd
import math
import re


class SentimentAnalyzer:
    """
    Hotel review sentiment analyzer

    Args:
        df:         Input DataFrame that contains hotel reviews
        review_col: Column corresponding to the reviews
        truth_col:  Column corresponding to the ground truth
        copy:       Copy flag
        analyzer:   Function to output sentiment score
        stop_words: Stop word collections
        pos_tag:    Part-of-speech tagger
        parser:     Regex parser to parse a chunk tree
        lemmatizer: Lemmatizer to apply lemmatization

    """
    # Default grammar to parse tree
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """

    def __init__(self,
                 df,
                 review_col,
                 truth_col,
                 copy=True,
                 analyzer=None,
                 stop_words=None,
                 pos_tag=None,
                 parser=None,
                 lemmatizer=None):

        # DataFrame stuffs
        self.df = df.copy() if copy else df
        self.review_col = review_col
        self.truth_col = truth_col

        # NLP stuffs
        self.analyzer = self.vader if analyzer is None else analyzer
        self.stop_words = stopwords.words('english') if stop_words is None else stop_words
        self.pos_tag = PerceptronTagger().tag if pos_tag is None else pos_tag
        self.parser = RegexpParser(self.grammar) if parser is None else parser
        self.lemmatizer = WordNetLemmatizer if lemmatizer is None else lemmatizer

    @staticmethod
    def vader(x):
        sid = SentimentIntensityAnalyzer()
        return sid.polarity_scores(x)['compound']

    def evaluate(self):
        """
        Evaluate hotel reviews with a given sentiment measure

        Returns:
            A DataFrame that contains a column of sentiment scores
        """

        self.df['Score'] = self.df[self.review_col].apply(self.analyzer)

        return self.df

    def _filter(self, word):
        """Checks conditions for acceptable word: length, stopword."""

        return bool(2 <= len(word) <= 40 and word.lower() not in self.stop_words)

    def _normalize(self, word):
        """Normalises words to lowercase and lemmatizes it."""

        return self.lemmatizer.lemmatize(word.lower())

    @staticmethod
    def _leaves(tree):
        """Finds NP (noun phrase) leaf nodes of a chunk tree."""

        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP' or
                                                      t.label() == 'JJ' or
                                                      t.label() == 'RB'):
            yield subtree.leaves()

    def _gen_terms(self, tree):
        """Generate a phrase one at a time"""

        for leaf in self._leaves(tree):
            terms = [self._normalize(w) for w, _ in leaf if self._filter(w)]
            # Phrase only
            if len(terms) > 1:
                yield terms

    @staticmethod
    def _flatten(phrases):
        """Flatten phrase lists to get tokens for analysis"""

        return [' '.join(phrase) for phrase in phrases]

    def tokenize(self, text, term_type='w'):
        """
        Split text into tokens by the specified type

        Args:
            text: Input text
            term_type: word or noun phrase

        Returns:
            tokens: A list of tokens
        """

        if term_type == 'w':
            tokens = [self._normalize(word)
                      for word in re.findall(r'\w+', str(text))
                      if self._filter(word)]
        elif term_type == 'np':
            tags = self.pos_tag(re.findall(r'\w+', str(text)))
            parsed = self.parser.parse(tags)
            tokens = self._flatten([phrase for phrase in self._gen_terms(parsed)])
        else:
            raise Exception("Unknown specified term_type '{}'".format(term_type))

        return tokens

    def get_topk_terms(self, k, label_val, term_type='w'):
        """
        Retrieve top-k terms, e.g., words or noun phrases.

        Args:
            k:         Top-k
            label_val: Either 'Positive' or 'Negative'
            term_type: Either 'w' or 'np' as word or noun-phrase

        Returns:
            Top-k frequent terms (A list of tuples)
        """

        counter = Counter()
        label_reviews = self.df.loc[self.df[self.truth_col] == label_val][self.review_col]
        for review in label_reviews:
            terms = self.tokenize(review, term_type=term_type)
            counter.update(terms)

        return counter.most_common(k)

    def get_coocurr(self, k, label_val, term_type='w', **kwargs):
        """
        Calculate co-occurrences of terms specified with reviews

        Args:
            k:         Top-k
            label_val: Either 'Positive' or 'Negative'
            term_type: Either 'w' or 'np' as word or noun-phrase
            topk:      Top-k terms

        Returns:
            A DataFrame with co-occurrences of terms
        """

        topk = kwargs.get('topk', self.get_topk_terms(k, label_val, term_type=term_type))
        topk_terms = [term for term, _ in topk]

        coocurr = []
        for review in self.df[self.review_col]:
            counter = Counter(self.tokenize(review, term_type=term_type))
            coocurr.append([1 if counter[term] else 0 for term in topk_terms])

        coocurr_df = pd.DataFrame(coocurr)
        coocurr_df.column = topk_terms

        return coocurr_df

    def get_mi(self, k, label_val, term_type='w', **kwargs):
        """
        Calculate mutual information score

        Args:
            k:          Top-k
            label_val:  Either 'Positive' or 'Negative'
            term_type:  Either 'w' or 'np' as word or noun-phrase
            topk:       Top-k terms
            coocurr_df: Co-occurrence DataFrame

        Returns:
            mi_df: A DataFrame that contains mutual information score for each term
        """

        topk = kwargs.get('topk', self.get_topk_terms(k, label_val, term_type=term_type))
        topk_terms = [term for term, _ in topk]

        coocurr_df = kwargs.get('coocurr', self.get_coocurr(k, label_val, term_type='w', topk=topk))

        labels_true = self.df[self.truth_col]
        mi_scores = [metrics.mutual_info_score(labels_true, coocurr_df[term]) for term in topk_terms]
        mi_df = pd.DataFrame(mi_scores)
        mi_df.columns = topk_terms

        return mi_df



