import os, argparse, ConfigParser, logging
from datetime import datetime as dt
import csv
import pandas as pd
import numpy as np
import gensim
import codecs
import pprint
import seaborn as sns
import nltk

import sentimentprocess as sp
import ngrams as ng
import tfidf as tt


DEFAULT = 'DEFAULT'
SCRIPT_DIR = os.path.dirname(os.path.realpath('__file__'))
config_file = SCRIPT_DIR + '/config.ini'
config = ConfigParser.ConfigParser()
config.read(config_file)
    
LOG_FILE = config.get(DEFAULT,'log_file')
LOG_LEVEL = config.get(DEFAULT,'log_level')
LOG_TS = dt.now().strftime("-%Y-%m-%d-%H")

if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)

logger = logging.getLogger('gchat_sms_analysis')
handler = logging.FileHandler(LOG_FILE + LOG_TS + '.log')
formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(funcName)s %(message)s'
        )
log_level = logging.ERROR if LOG_LEVEL == 'ERROR' else \
            logging.WARNING if LOG_LEVEL == 'WARNING' else \
            logging.INFO if LOG_LEVEL == 'INFO' else\
            logging.DEBUG

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(log_level)



def main():

    """
    Decides which method to pursue with the infile data. 

    """

    args = get_arguments()

    data = Data

    if args.infile_basic and args.outfile:
        print 'Processing input file: {0} for basic descriptives.'.format(args.infile_basic)
        process_descriptives(args.infile_basic, outfile=args.outfile)
        print 'Saved descriptives and by day descriptives.'

    if args.infile_ngrams:
        print 'Processing input file: {0} for {1}-gram descriptives'.format(args.infile_ngrams, args.num_grams)
        process_ngrams(args.infile_ngrams, args.num_grams, outfile=args.outfile)
        # add desired timestamp as args when using plot tfidf
        print 'Finished creating ngram descriptves.'

    if args.infile_tfidf:
        print 'Processing input file: {0} for tfidf descriptives'.format(args.infile_tfidf)
        data.df_tfidf = process_tfidf(args.infile_tfidf, outfile=args.outfile)
        data.df_tfidf = get_tfidf_descrps(data.df_tfidf)
        # add desired timestamp as args when using plot_tfidf
        print 'Finished creating tfidf descriptives.'


def plot_tfidf(data, timestamp, author):

    """
    Prints high and low TFIDF scores, plots the spectrum of high, medium, and low TFIDFs in a simple histogram.

    """

    # data.index = [str(dt.date(x)) for x in data.index]
    plot_data = data[data['author']==author].loc[timestamp]

    pprint.pprint(plot_data.tfidf_high)
    pprint.pprint(plot_data.tfidf_low)

    for i in range(len(plot_data.tfidf_list)):
        sns.set(rc={'figure.figsize': (30, 10)}, style='whitegrid')
        sns_plot = sns.barplot(x=[x[0] for x in plot_data.tfidf_list[i]], y=[x[1] for x in plot_data.tfidf_list[i]], palette='Blues_d')
        sns_plot.set_xticklabels([x[0] for x in plot_data.tfidf_list[i]], rotation=30)
        sns_plot.set(xlabel='Word', ylabel='TFIDF Score')
        fig = sns_plot.get_figure()
        fig.savefig('plot_segment_{0}'.format(i))
        fig.clf()
        
    print 'Finished.'


def get_tfidf_descrps(data):

    """
    Calculates TFIDF scores for each week and author.

    """

    # rows from here on out
    list_weekly_logs = data.message.tolist()
    set_weekly_logs = [set(sublist) for sublist in list_weekly_logs]
    full_vocabulary = set([word for sublist in list_weekly_logs for word in sublist])

    tf = [{word: tt.get_tf(word, doc) for word in doc} for doc in list_weekly_logs]
    idf = tt.get_idf(full_vocabulary, set_weekly_logs)

    tfidf_list = [{word: tf_score * idf[word] for word, tf_score in word_dict.iteritems()} for word_dict in tf]
    tfidf_high = [sorted(word_dict.iteritems(), key=lambda x: -x[1])[:20] for word_dict in tfidf_list]
    tfidf_low = [sorted(word_dict.iteritems(), key=lambda x: -x[1])[-20:] for word_dict in tfidf_list]
    tfidf_list = [chunks(sorted(word_dict.iteritems(), key=lambda x: -x[1]), 3) for word_dict in tfidf_list]
    
    # will be deprecated
    data['tfidf_list'] = tfidf_list
    data['tfidf_high'] = tfidf_high
    data['tfidf_low'] = tfidf_low

    return data

    
def chunks(full_list, num_chunks):

    """
    Yield successive n-equally sized chunks from a full list.

    """

    chunked_list = []
    chunk_size = len(full_list)/num_chunks
    for i in range(num_chunks):
        chunked_list.append(full_list[int(round(i*chunk_size)):int(round((i+1)*chunk_size))])

    return chunked_list


def process_tfidf(filename, outfile=None):

    """
    Processes raw data into tf-idf metrics.

    """
    
    data = load_data(filename)

    data = [sp.process_timestamp(row) for row in data[1:]]
    data = [' ' if x is None else x for x in data]
    data = [sp.remove_weblinks(row) for row in data]
    data = [sp.modify_characters(row) for row in data]
    data = [sp.remove_stoppage(row) for row in data]

    data = pd.DataFrame(data, columns=['author', 'message', 'timestamp']).set_index('timestamp')
    data = data.groupby([pd.TimeGrouper(freq='7D'), 'author'])['message'].sum().reset_index().set_index('timestamp')

    return data


            
            


    

def process_ngrams(filename, num_grams, outfile=None):

    """
    Processes raw data into ngram frequency metrics.

    """

    data = Data()
    data.df_grams = load_data(filename)

    # bigram and trigram pre-processing step is different
    data.df_grams = [sp.process_timestamp(row) for row in data.df_grams[1:]]
    data.df_grams = [' ' if x is None else x for x in data.df_grams]
    data.df_grams = [sp.remove_weblinks(row) for row in data.df_grams]
    data.df_grams = [sp.modify_characters(row) for row in data.df_grams]
    data.df_grams = pd.DataFrame(data.df_grams, columns=['author', 'message', 'timestamp']).set_index('timestamp')
    data.df_grams = (
            data.df_grams.groupby([pd.TimeGrouper(freq='D'), 'author'])['message']
            .apply(lambda x: x.tolist())
            .reset_index()
            .set_index('timestamp')
        )

    # bi/tri grams by quarter, however we lose out on the person's thoughts
    data.df_grams = data.df_grams.groupby([pd.TimeGrouper(freq='3M')])['message'].sum().reset_index()

    if num_grams == 2:
        data.df_grams.message = data.df_grams.message.apply(ng.gen_bigrams)
        # print data.df_grams.message.iloc[0]
        
    if num_grams == 3:
        data.df_grams.message = data.df_grams.message.apply(ng.gen_trigrams)
        # print data.df_grams.message.iloc[0]

    data.df_grams.message = data.df_grams.message.apply(lambda x: [str(' '.join(entry)) for entry in x])

    # still might have to combine into weekly bigrams to get legit results?
    # data.df_grams.message = data.df_grams.message.apply(lambda x: nltk.FreqDist(x))
    # data.df_grams.message = data.df_grams.message.apply(lambda x: dict(x))


    # run through tf idf function

    # graph function? to graph frequency distributions for quarterly aggregations




def process_descriptives(filename, outfile=None):

    """
    Processes sentiment data.

    """
    
    data = Data()
    data.df = load_data(filename)

    header = data.df[0] + ['word_count', 'seconds_idle']
    data.df = [sp.process_timestamp(row) for row in data.df[1:]]
    data.df = [' ' if x is None else x for x in data.df]
    data.df = [sp.remove_weblinks(row) for row in data.df]
    data.df = [sp.modify_characters(row) for row in data.df]
    data.df = [sp.remove_stoppage(row) for row in data.df]
    data.df = [sp.get_word_count(row) for row in data.df]
    data.df = sp.get_seconds_idle(data.df)
    data.df = pd.DataFrame(data.df, columns=header)
    data.df = data.df.set_index('timestamp')
    data.df_by_day = sp.get_day_metrics(data.df)

    if outfile is not None:
        data.df.to_csv(outfile + '_data.csv')
        data.df_by_day.to_csv(outfile + '_data_by_day.csv')


def load_data(filename):

    """
    @input: filename
    @action: loads file as list of lists (rows)
    @output: returns data

    """
    
    data = read_csv(filename)
    
    return [row[1:] for row in data]


class Data(object):
    
    """
    The data class.

    """
    
    def __init__(self):
        
        self.df = None
        self.df_by_day = None
        self.df_grams = None
        self.df_tfidf = None
        
        
def write_csv(datalist, outfile, delimiter=',', quotechar='"'):
    
    """
    @input: list of lists (rows)
    @action: writes into csv
    @output: dataframe in csv
    
    """
    
    with open(outfile, 'wb') as csvfile:
        row_writer = csv.writer(
            csvfile, 
            delimiter=delimiter, 
            quotechar=quotechar,
            quoting=csv.QUOTE_MINIMAL
        )
        for row in datalist:
            row_writer.writerow(row)


def read_csv(filename, delimiter=',', quotechar='"'):
    
    """
    @input: csv file
    @action: reads the csv in by rows
    @output: list of lists (rows) 
    
    """
    
    results = []
    with open(filename, 'rU') as csvfile:
        csv_reader = csv.reader(
            csvfile, 
            delimiter=delimiter,
            quotechar=quotechar,
            quoting=csv.QUOTE_MINIMAL
        )
        results = [row for row in csv_reader]

    return results


def get_arguments():

    """
    Gets the arguments used for this script
    """

    parser = argparse.ArgumentParser(
            description='Google Takeout descriptive stats and data munge.'
        )
    
    parser.add_argument(
            '-basic',
            action='store',
            dest='infile_basic',
            help='Processed Google Takeout JSON data',
            required=False
        )

    parser.add_argument(
            '-ngram',
            action='store',
            dest='infile_ngrams',
            help='Processed Google Takeout JSON data',
            required=False
        )

    parser.add_argument(
            '-n',
            nargs='?',
            const=2,
            type=int,
            dest='num_grams',
            default=2,
            help='Number of grams to analyze',
            required=False
        )

    parser.add_argument(
            '-tfidf',
            action='store',
            dest='infile_tfidf',
            help='Processed Google Takeout JSON data',
            required=False
        )

    parser.add_argument(
            '-o',
            nargs='?',
            default=None,
            action='store',
            dest='outfile',
            help='google takeout descriptives',
            required=False
        )

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    main()
