import pandas as pd
import nltk
import string
import re
from datetime import datetime as dt



def get_day_metrics(df):
    
    """
    @input: dataframe
    @action: creates by day metrics
    @output: by day dataframe

    """
    
    words_by_day = (
        df.groupby([pd.TimeGrouper(freq='D'), 'author'])['word_count']
            .sum()
            .reset_index()
            .rename(columns={0: 'num_words'})
        ).set_index('timestamp')
    
    messages_by_day = (
        df.groupby([pd.TimeGrouper(freq='D'), 'author'])
            .size()
            .reset_index()
            .rename(columns={0: 'num_messages'})
        ).set_index('timestamp')

    total_messages = (
        messages_by_day.groupby(pd.TimeGrouper(freq='D'))
            .sum()
            .rename(columns={'num_messages': 'total_messages'})
        )

    all_by_day = (
        words_by_day
            .reset_index()
            .merge(messages_by_day.reset_index(), on=['timestamp', 'author'])
            .set_index('timestamp')
            .merge(total_messages, left_index=True, right_index=True)
        )

    all_by_day['%_response'] = (
        all_by_day.num_messages/all_by_day.total_messages
        )

    return all_by_day


def get_seconds_idle(data):
    
    """
    @input: data
    @action: finds idle time of each message
    @output: data with idle times
    
    """
    
    for i in range(len(data)-1):
        data[i].append((data[i + 1][2] - data[i][2]).seconds)

    return data


def get_word_count(row):
    
    """
    @input: dataframe
    @action: obtain message length of messages
    @output: dataframe with additional column with length of messages.
    
    """
    
    row.append(len(row[1]))
    
    return row


def remove_stoppage(row):
    
    """
    @input: row message word list (index position 1) with stopwords
    @action: removes punctuation and stopwords
    @output: row with message word list without stopwords
    
    """
    
    row[1] = [
            word for word in row[1]
            if word not in (
                    map(lambda x: x, string.punctuation) +
                    nltk.corpus.stopwords.words('english')
                )
        ]
    
    return row


def modify_characters(row):
    
    """
    @input: row with non-alphanumeric characters, uppercases, whitespaces 
    in the message (index position 1)
    @action: removes non-alpha characters, switches characters to lowercase, 
    removes whitespaces from message body
    @output: row with none of the above, returns message as a list of words
    
    """
    
    row[1] = re.sub('[^a-zA-z]', ' ', re.sub('\'', '', row[1])).lower().split()
    
    return row


def remove_weblinks(row):
    
    """
    @input: row with weblinks in the message (index position 1)
    @action: removes weblinks from message body
    @output: row with messages without weblinks
    
    """
    
    row[1] = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|'
                    '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', row[1]) 

    return row


def process_timestamp(row):

    """
    @input: row with timestamp in index position 2
    @action: converts string time to datetime object
    @output: row with timedelta timestamp

    """
    
    row[2] = dt.strptime(str(row[2]), '%Y-%m-%d %H:%M:%S')
    
    return row



if __name__ == "__main__":

    print 'Processing script for Google JSON takeout data.'
