from enum import Enum

class Ex1_Headers(Enum):
    CREDITABILITY = 'Creditability'
    BALANCE = 'Account Balance'
    CREDIT_DURATION = 'Duration of Credit (month)'
    PREVIOUS_CREDIT_STATUS = 'Payment Status of Previous Credit'
    PURPOSE = 'Purpose'
    CREDIT_AMOUNT = 'Credit Amount'
    VALUE_SAVINGS = 'Value Savings/Stocks'
    EMPLOYMENT_LENGTH = 'Length of current employment'
    INSTALMENT_PERCENT = 'Instalment per cent'
    SEX_AND_MARRIAGE = 'Sex & Marital Status'
    GUARANTORS = 'Guarantors'
    CURRENT_ADDRESS_DURATION = 'Duration in Current address'
    MOST_VALUABLE_ASSET = 'Most valuable available asset'
    AGE = 'Age (years)'
    CONCURRENT_CREDITS = 'Concurrent Credits'
    APARTMENT_TYPE = 'Type of apartment'
    CREDITS_AT_BANK = 'No of Credits at this Bank'
    OCCUPATION = 'Occupation'
    DEPENDENTS = 'No of dependents'
    PHONE = 'Telephone'
    FOREIGNER = 'Foreign Worker'

class Tennis_Headers(Enum):
    FORECAST = 'Pronostico'
    TEMP = 'Temperatura'
    HUMIDITY = 'Humedad'
    WIND = 'Viento'
    PLAYS = 'Juega'

class Ex2_Headers(Enum):
    TITLE = 'Review Title'
    TEXT = 'Review Text'
    WORDCOUNT = 'wordcount'
    TITLE_SENTIMENT = 'titleSentiment'
    TEXT_SENTIMENT = 'textSentiment'
    STAR_RATING = 'Star Rating'
    SENTIMENT_VALUE = 'sentimentValue'
    # Custom Headers
    STAR_RATING_NORM = 'Star Rating Normalized'
    ORIGINAL_ID = "Original Id"

class Ex2_Modes(Enum):
    WEIGHTED = 'weighted'
    SIMPLE = 'simple'

class Ex2_Run(Enum):
    SOLVE = 'solve'
    ANALYZE = 'analyze'

class Ex2_Title_Sentiment(Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'

# Number of slots divided
EX1_DIVISION = 10
EX2_DIVISION = 10
