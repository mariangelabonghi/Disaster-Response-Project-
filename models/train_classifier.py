"""
Input:
- Path to SQLite destination database
- Path to pickle file name where ML model needs to be saved
To run the script:
python train_classifier.py <path to sqllite  destination db> <path to the pickle file>
"""
import sys
import pandas as pd
import sqlalchemy as sa
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def load_data(database_filepath):
    """
    Load data from database and return
    X: column message
    y: categories
    category_names: names of the categories
    """
    engine = sa.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns # This will be used for visualization purpose
    return X, y, category_names

def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize text
    Input:
        text -> Text message to be tokenized
    Output:
        words -> List of tokens extracted from the input text
    """
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)

    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # tokenize text
    tokens = word_tokenize(text)
    #Lemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    The class StartingVerbExtractor extracts the starting verb of a sentence,
    creating a new feature for the ML classifier
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build Pipeline function returns a ML pipeline
    that process text messages and apply a classifier.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {
        'clf__estimator__n_estimators': [25, 50, 100],
        'clf__estimator__max_leaf_nodes': [2, 3, 4]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model

def evaluate_model(model, X_test, y_test, category_names):
    """
    This function applies a ML pipeline to a test set
    and prints out the model performance
    """
    y_pred = model.predict(X_test)
    for i, j in enumerate(category_names):
        classification_report(y_test[j], y_pred[:,i])

def save_model(model, model_filepath):
    '''
    Save trained model as Pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
