# Second Year Project
# Introduction to Natural Language Processing and Deep Learning

This repository contains the work for a three phase project in Natural Language Processing. Phase 1 is the development of an NLP model for the purpose of sentiment classification of Amazon music reviews. Stage 2 involved creating edge cases that would challenge the model that was employed in stage one, as well as the predictions of these hard cases. Stage 3 involves a novel study related to sentiment analysis in a cross-domain setting.

## Getting Started

The language used in this project is Python Version 3.8.2

### Prerequisites

Please see requirements.txt for libraries and versions

## How to Run everything:
Most up-to-date version can be found in 'The Big How to Reproduce Our Findings Guide'.txt

Environment requirements:
All required libraries/versions etc can be found in the requirements.txt

General syntax for code
Train/finetune model: code/baseline.py 0/1 training_data_path val_data_path model_or_None new_model_name

Creating Datasets:
The interim datasets were created by loading the JSON files, and outputting CSVs containing only the text and the label.
Required files:
code/data_prep.ipynb # This is a jupyter notebook. When run, it will output music_train.csv, music_dev.csv and music_test.csv to the data/interim folder. The raw data was already split into train/dev/test.

code/corpus_load.py
Run via:
python3 code/corpus_load.py Arts_Crafts_and_Sewing.json.gz sew
python3 code/corpus_load.py Video_Games.json.gz games
Will output to data/interim 3 csv files for each: train, dev & test.

####################################################################################
Need to add the bit about creating the dissimilar/random csvs here!!!
####################################################################################


Creating Models:

Baseline Model:
Required files:
code/baseline.py
data/interim/music_train.csv
data/interim/music_dev.csv

Run via:
python3 code/baseline.py 1 'data/interim/music_train.csv' 'data/interim/music_dev.csv' None 'base'

This will output a pickled model which can be found at: code/models/model_base.pkl

Finetune experiments with new data:
Required files:
code/baseline.py
data/dissimilar/games*.csv (4 files)
data/dissimilar/sew*.csv (4 files)
data/random/games_*.csv(4 files)
data/random/sew_*.csv(4 files)
data/random/games_res_*.csv(4 files)
data/random/sew_res_*.csv(4 files)

Run via:

Selected:
python3 code/baseline.py 0 'data/dissimilar/games10.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_00010cp'
python3 code/baseline.py 0 'data/dissimilar/games100.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_00100cp'
python3 code/baseline.py 0 'data/dissimilar/games1000.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_01000cp'
python3 code/baseline.py 0 'data/dissimilar/games10000.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_10000cp'
python3 code/baseline.py 0 'data/dissimilar/sew10.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_00010cp'
python3 code/baseline.py 0 'data/dissimilar/sew100.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_00100cp'
python3 code/baseline.py 0 'data/dissimilar/sew1000.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_01000cp'
python3 code/baseline.py 0 'data/dissimilar/sew10000.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_10000cp'

Randomised:
python3 code/baseline.py 0 'data/random/games_00010.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_00010ra'
python3 code/baseline.py 0 'data/random/games_00100.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_00100ra'
python3 code/baseline.py 0 'data/random/games_01000.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_01000ra'
python3 code/baseline.py 0 'data/random/games_10000.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_10000ra'
python3 code/baseline.py 0 'data/random/sew_00010.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_00010ra'
python3 code/baseline.py 0 'data/random/sew_00100.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_00100ra'
python3 code/baseline.py 0 'data/random/sew_01000.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_01000ra'
python3 code/baseline.py 0 'data/random/sew_10000.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_10000ra'

Balanced Randomised:
python3 code/baseline.py 0 'data/random/games_res_00010.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_00010es'
python3 code/baseline.py 0 'data/random/games_res_00100.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_00100es'
python3 code/baseline.py 0 'data/random/games_res_01000.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_01000es'
python3 code/baseline.py 0 'data/random/games_res_10000.csv' 'data/interim/games_val.csv' code/models/model_base.pkl 'games_10000es'
python3 code/baseline.py 0 'data/random/sew_res_00010.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_00010es'
python3 code/baseline.py 0 'data/random/sew_res_00100.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_00100es'
python3 code/baseline.py 0 'data/random/sew_res_01000.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_01000es'
python3 code/baseline.py 0 'data/random/sew_res_10000.csv' 'data/interim/sew_val.csv' code/models/model_base.pkl 'sew_10000es'

Each line will output a pickled model into the code/models folder.


Create Predictions:
Required files:
code/test.py
code/model_run_v2.py
code/predict_max_v2.py
It is required that the model_base.pkl from above, as well as a
10, 100, 1000, 10000 model exists for each domain being tested.

Run via:
python3 code/test.py sew cp
python3 code/test.py games cp

python3 code/test.py sew ra
python3 code/test.py games ra

python3 code/test.py sew es
python3 code/test.py games es

These will run the test datasets against each model within each catagory, and output for each a probabilities file and predictions file, in data/probabilities and data/predictions respectively. 

Get Metrics:
Required files:
code/metrics.py
data/predictions/*all the prediction files from above!*

Run Via:
python3 code/metrics.py games 00000ba 00010cp 00100cp 01000cp 10000cp 00010ra 00100ra 01000ra 10000ra 00010es 00100es 01000es 10000es > report/metrics/games_mixed_metrics.txt

python3 code/metrics.py sew 00000ba 00010cp 00100cp 01000cp 10000cp 00010ra 00100ra 01000ra 10000ra 00010es 00100es 01000es 10000es > report/metrics/sew_mixed_metrics.txt

It it is not required to pipe it to the .txt file, but this is more human readable if you just want to look quickly, it will print to the terminal otherwise.
It will also output the same information into a CSV file in the report/metrics folder, with the headers:
['domain', 'trial_type', 'add_data', 'correctly_predicted', 'incorrectly_predicted', 'total_predicted_positives', 'ground_truth_positives', 'TP', 'TN', 'FP', 'FN', 'accuracy', 'precision', 'recall', 'f1']

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used

## Contributing

Group 12

* Danielle Dequin ddeq@itu.dk
* Chrisanna Cornish ccor@itu.dk
* Sabrina Pereira sabf@itu.dk

## Authors

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under no such License - see the [LICENSE.md](LICENSE.md) file for details if we make one.

## Acknowledgments

* This [tutorial](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2) for the ReadMe template.
