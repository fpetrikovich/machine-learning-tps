from enum import Enum

class Ex2_Headers(Enum):
    TITLE = 'Review Title'
    TEXT = 'Review Text'
    WORDCOUNT = 'wordcount'
    TITLE_SENTIMENT = 'titleSentiment'
    TEXT_SENTIMENT = 'textSentiment'
    STAR_RATING = 'Star Rating'
    SENTIMENT_VALUE = 'sentimentValue'

class Ex2_Modes(Enum):
    WEIGHTED = 'weighted'
    SIMPLE = 'simple'

class Ex2_Title_Sentiment(Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'

# Number of slots divided
EX2_DIVISION = 10