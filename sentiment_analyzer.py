from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import PerceptronTagger
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import RegexpParser
from collections import Counter
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
        """Finds NP (noun phrase) leaf nodes of a tree chunk."""

        for st in tree.subtrees(filter=lambda t: t.label() in ['NP', 'JJ', 'RB']):
            yield st.leaves()

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
        Tokenize text by the specified type

        Args:
            text:      str. Input text
            term_type: str. Word or noun phrase, e.g., 'w' or 'np'

        Returns:
            tokens: A list of tokens
        """

        words = re.findall(r'\w+', str(text))

        if term_type == 'w':
            tokens = [self._normalize(word) for word in words if self._filter(word)]
        elif term_type == 'np':
            # Parsed tree based on grammar structure from POS tags
            parsed = self.parser.parse(self.pos_tag(words))
            tokens = self._flatten([phrase for phrase in self._gen_terms(parsed)])
        else:
            raise Exception("Unknown specified term_type '{}'".format(term_type))

        return tokens

    def get_topk_terms(self, k, label, term_type='w'):
        """
        Retrieve top-k terms (e.g., words or noun phrases).

        Args:
            k:         int. Top-k
            label:     str. Label, e.g., 'positive' or 'negative'
            term_type: str. Word or noun phrase, e.g., 'w' or 'np'

        Returns:
            Top-k frequent terms (A list of tuples)
        """

        counter = Counter()
        # Retrieve reviews labeled by the specified label_val
        label_reviews = self.df.loc[self.df[self.truth_col] == label][self.review_col]
        for review in label_reviews:
            terms = self.tokenize(review, term_type)
            counter.update(terms)

        return counter.most_common(k)

    def get_coocurr(self, k, label, term_type='w', **kwargs):
        """
        Calculate term co-occurrences w.r.t reviews

        Args:
            k:         int. Top-k
            label:     str. Label, e.g., 'positive' or 'negative'
            term_type: str. Word or noun phrase, e.g., 'w' or 'np'

            kwargs:
                topk: list. Top-k word counts

        Returns:
            A DataFrame with term co-occurrences
        """

        topk = kwargs.get('topk', self.get_topk_terms(k, label, term_type))
        topk_terms = [term for term, _ in topk]

        coocurr = []
        for review in self.df[self.review_col]:
            tokens = set(self.tokenize(review, term_type))
            coocurr.append([1 if term in tokens else 0 for term in topk_terms])

        coocurr_df = pd.DataFrame(coocurr)
        coocurr_df.column = topk_terms

        return coocurr_df

    def get_mi(self, k, label, term_type='w', **kwargs):
        """
        Calculate mutual information score for each term w.r.t ground truths

        Args:
            k:         int. Top-k
            label:     str. Label, e.g., 'positive' or 'Negative'
            term_type: str. Word or noun phrase, e.g., 'w' or 'np'

            kwargs:
                topk:       list. Top-k word counts
                coocurr_df: DataFrame. Term co-occurrences

        Returns:
            mi_df: A DataFrame that contains MI score for each term
        """

        topk = kwargs.get('topk', self.get_topk_terms(k, label, term_type))
        topk_terms = [term for term, _ in topk]

        coocurr_df = kwargs.get('coocurr', self.get_coocurr(k, label, term_type, topk=topk))

        labels_true = self.df[self.truth_col]

        mi = [[term, metrics.mutual_info_score(labels_true, coocurr_df[term])]
              for term in topk_terms]

        mi_df = pd.DataFrame(mi)
        mi_df.columns = ['Term', 'MI']

        return mi_df

    def get_pmi(self, k, label, term_type='w', **kwargs):
        """
        Calculate point-wise mutual information score for each term w.r.t the input label

        Args:
            k:         int. Top-k
            label:     str. Label, e.g., 'positive' or 'Negative'
            term_type: str. Word or noun phrase, e.g., 'w' or 'np'

            kwargs:
                topk:       list. Top-k word counts
                coocurr_df: DataFrame. Term co-occurrences

        Returns:
            pmi_df: A DataFrame that contains PMI score for each term

        """

        topk = kwargs.get('topk', self.get_topk_terms(k, label, term_type))
        topk_terms = [term for term, _ in topk]

        coocurr_df = kwargs.get('coocurr', self.get_coocurr(k, label, term_type, topk=topk))

        # Total number of reviews
        num_reviews = self.df.shape[0]
        # Total number of input labels
        is_label = self.df[self.truth_col] == label
        py = pd.Series(is_label).sum()

        pmi = []
        for term in topk_terms:
            # Total number of this term's occurrences
            px = coocurr_df[term].sum()
            # Total number of this term's occurrences as observed along with the label
            pxy = self.df[(coocurr_df[term] == 1) & is_label].shape[0]

            # log zero handler
            pxy += 1e-4 if pxy == 0 else pxy

            pmi.append([term, math.log2(num_reviews * pxy / (px * py))])

        pmi_df = pd.DataFrame(pmi)
        pmi_df.columns = ['Term', 'PMI']

        return pmi_df
