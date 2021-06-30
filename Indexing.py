data = 'E:/THESIS/scibert/CORD/2021-03-29/metadata.csv'

import pandas as pd
import numpy as np
import os
from collections import defaultdict

import whoosh
from whoosh.qparser import *
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh.analysis import StandardAnalyzer
from whoosh import index

from whoosh.analysis import Tokenizer, Token
from whoosh import highlight
from IPython.core.display import display, HTML

df = pd.read_csv(data, low_memory=False)



# (497906, 19)
# Index(['cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id',
#        'license', 'abstract', 'publish_time', 'authors', 'journal', 'mag_id',
#        'who_covidence_id', 'arxiv_id', 'pdf_json_files', 'pmc_json_files',
#        'url', 's2_id'],
#       dtype='object')

import en_core_sci_lg

# medium model
nlp = en_core_sci_lg.load(disable=["tagger", "parser", "ner","lemmatizer"])
nlp.max_length = 2000000

# New stop words list
customize_stop_words = [
    'doi', 'preprint', 'copyright', 'org', 'https', 'et', 'al', 'author', 'figure', 'table', 'cord_uid', 'sha'
    'rights', 'reserved', 'permission', 'use', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI',
    '-PRON-', 'usually',
    r'\usepackage{amsbsy', r'\usepackage{amsfonts', r'\usepackage{mathrsfs', r'\usepackage{amssymb', r'\usepackage{wasysym',
    r'\setlength{\oddsidemargin}{-69pt',  r'\usepackage{upgreek', r'\documentclass[12pt]{minimal'
]

# Mark them as stop words
for w in customize_stop_words:
    nlp.vocab[w].is_stop = True

def my_spacy_tokenizer(sentence):
    # lowercase lemma, startchar and endchar of each word in sentence
    return [(word.lemma_.lower(), word.idx, word.idx + len(word)) for word in nlp(sentence)
            if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]


class SpacyTokenizer(Tokenizer):
    """
    Customized tokenizer for Whoosh
    """

    def __init__(self, spacy_tokenizer):
        self.spacy_tokenizer = spacy_tokenizer

    def __call__(self, value, positions=False, chars=False,
                 keeporiginal=False, removestops=True,
                 start_pos=0, start_char=0, mode='', **kwargs):
        """
        :param value: The unicode string to tokenize.
        :param positions: Whether to record token positions in the token.
        :param chars: Whether to record character offsets in the token.
        :param start_pos: The position number of the first token. For example,
            if you set start_pos=2, the tokens will be numbered 2,3,4,...
            instead of 0,1,2,...
        :param start_char: The offset of the first character of the first
            token. For example, if you set start_char=2, the text "aaa bbb"
            will have chars (2,5),(6,9) instead (0,3),(4,7).
        """

        assert isinstance(value, str), "%r is not unicode" % value

        spacy_tokens = self.spacy_tokenizer(value)

        t = Token(positions, chars, removestops=removestops, mode=mode)

        for pos, spacy_token in enumerate(spacy_tokens):
            t.text = spacy_token[0]
            if keeporiginal:
                t.original = t.text
            t.stopped = False
            if positions:
                t.pos = start_pos + pos
            if chars:
                t.startchar = start_char + spacy_token[1]
                t.endchar = start_char + spacy_token[2]
            yield t


# get hardcoded schema for the index
def get_search_schema(analyzer=StandardAnalyzer()):
    schema = Schema(paper_id=ID(stored=True),
                    doi=ID(stored=True),
                    authors=TEXT(analyzer=analyzer),
                    journal=TEXT(analyzer=analyzer),
                    title=TEXT(analyzer=analyzer, stored=True),
                    abstract=TEXT(analyzer=analyzer, stored=True),
                    source_x=TEXT(analyzer=analyzer)
                    #                     body_text = TEXT(analyzer=analyzer)
                    )
    return schema


def add_documents_to_index(ix, df):
    # create a writer object to add documents to the index
    writer = ix.writer()

    # now we can add documents to the index
    for _, doc in df.iterrows():
        writer.add_document(paper_id=str(doc.cord_uid),
                            doi=str(doc.doi),
                            authors=str(doc.authors),
                            journal=str(doc.journal),
                            title=str(doc.title),
                            abstract=str(doc.abstract),
                            source_x=str(doc.source_x)
                            #                             body_text = str(doc.body_text)
                            )

    writer.commit()

    return


def create_search_index(search_schema):
    if not os.path.exists('indexdir'):
        os.mkdir('indexdir')
        ix = index.create_in('indexdir', search_schema)
        add_documents_to_index(ix, df)
    else:
        # open an existing index object
        ix = index.open_dir('indexdir')
    return ix

my_analyzer = SpacyTokenizer(my_spacy_tokenizer)
search_schema = get_search_schema(analyzer=my_analyzer)
ix = create_search_index(search_schema)
parser = MultifieldParser(['title', 'abstract'], schema=search_schema, group=OrGroup.factory(0.9))
parser.add_plugin(SequencePlugin())

df.set_index('cord_uid', inplace=True)


def search(query, lower=1950, upper=2020, only_covid19=False, kValue=5):
    query_parsed = parser.parse(query)
    with ix.searcher() as searcher:
        results = searcher.search(query_parsed, limit=None)
        output_dict = defaultdict(list)
        for result in results:
            output_dict['cord_uid'].append(result['cord_uid'])
            output_dict['score'].append(result.score)

    search_results = pd.Series(output_dict['score'], index=pd.Index(output_dict['cord_uid'], name='cord_uid'))
    search_results /= search_results.max()

    file = r'E:/THESIS/scibert/CORD/2021-03-29/metadata.csv'
    headers = []
    idx = -1
    contents = {}
    time = {}
    with open(file, 'r', encoding='utf=8') as f:
        for i in range(30000):
            line = f.readline()
            line = line[:-1]
            line = line.replace(',"""', ',"|||')
            line = line.replace('""",', '|||",')
            line = line.replace('""', "|||")
            items = line.split('"')
            temp = []
            for i in range(len(items)):
                if i % 2 == 0:
                    if items[i].startswith(','):
                        items[i] = items[i][1:]
                    for item in items[i][:-1].split(','):
                        temp.append(item)
                else:
                    temp.append(items[i].replace("|||", '"'))
            items = temp
            while len(items) < len(headers):
                items.append('')
            if len(headers) == 0:
                for i in range(len(items)):
                    headers.append(items[i])
                    if items[i] == 'publish_time':
                        idx = i
                continue
            id = True
            x = []
            temp = ''
            for item in items:
                if id:
                    id = False
                    temp = item
                    continue
                x.append(item)
            if len(items[idx]) > 0:
                time[temp] = int(items[idx][0:4])
                contents[temp] = x
    dataset = {}
    for key in time.keys():
        date = time[key]
        if date >= 2000 and date <= 2020:
            dataset[key] = contents[key]

    fd = pd.DataFrame(data=time, index=[0])
    # df = pd.DataFrame.transpose(df)
    fd = fd.iloc[0]
    relevant_time = fd.between(lower, upper)


    if only_covid19:
        temp = search_results[relevant_time & df.is_covid19]

    else:
        temp = search_results[relevant_time]

    if len(temp) == 0:
        return -1

    # Get top k matches
    top_k = temp.nlargest(kValue)

    return top_k


def searchQuery(query, lower=1950, upper=2020, only_covid19=False,
                attr=['paper_id', 'title', 'abstract', 'url', 'authors'], kValue=3):
    search_results = search(query, lower, upper, only_covid19, kValue)

    if type(search_results) is int:
        return []
    # results = df.loc[search_results].reset_index()
    results = pd.merge(search_results.to_frame(name='similarity'), df, on='paper_id', how='left').reset_index()

    return results[attr].to_dict('records')


