# Disaster Response Project  

## by Mariangela Bonghi


## Description
This is the second project for the "Data Scientist" [Udacity](https://www.udacity.com) Nanodegree
in collaboration with Figure Eight that has provided pre-labeled tweets and text-messages from
real life disaster.

The task of the project is to repair this data with an ETL pipeline
and then use a Machine Learning Pipeline to build a supervised learning model.

Here below the main tasks:

- build an ETL pipeline to extract data, transform data and finally save them in a SQLite DB
- build a machine learning pipeline to classify text message in various categories
- Run a web app which can show model results in real time

## File Description:
- app : This folder contains the script "run.py" for the web app;
- data: This folder contains sample messages and categories csv files and the following files:
  - ETL Pipeline Preparation.ipynb: notebook for preparation of the script "process_data.py";
  - script "process_data.py": This script takes in input csv files containing message data and message  categories; it cleans the data and save the final dataset in a sqlLite DB database_filename;
- models: This folder contains the following files:  
  - ML Pipeline Preparation.ipynb: notebook for preparation of the script "train_classifier.py";
  - script "train_classifier.py": This script load data from the database, define the model, evaluate the model and finally save the ML model.

## How to...

- clone the project running:
git clone https://github.com/mariangelabonghi/Disaster-Response-Project-.git
- install the following libraries: Pandas, Numpy, sqlalchemy,re, nltk, sklearn, pickle
- Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

- Run the following command in the app's directory to run your web app.
    `python run.py`

- Go to http://0.0.0.0:3001/

## Acknowledgements
- "Data Scientist" [Udacity](https://www.udacity.com) Nanodegree: the project was developed for this course;
- Figure Eight that has provided pre-labeled tweets and text-messages from real life disaster.
