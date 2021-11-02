# Instruction for dataset "Alarm logs of industrial packaging machines"
## Metadata
### Authors
- Diego Tosato, ORCID, Galdi Srl
- Davide Dalle Pezze, 0000-0002-4741-1021, Università degli Studi di Padova
- Chiara Masiero, 0000-0003-1948-049X, Statwolf Data Science Srl
- Gian Antonio Susto, 0000-0001-5739-9639, Università degli Studi di Padova
- Alessandro Beghi, 0000-0003-2252-2179, Università degli Studi di Padova

### Category
Artificial Intelligence, IoT, Machine Learning
### Keywords
Industry 4.0; Industrial IoT; Machine Learning; Alarm sequences data
### File
*.csv (zip); *.json (zip); *.pickle (zip); *.npz (zip);

## Abstract
The advent of the Industrial Internet of Things (IIoT) has led to the availability of huge amounts of data, that can be used to train advanced Machine Learning algorithms to perform tasks such as <b>Anomaly Detection, Fault Classification and Predictive Maintenance</b>. Even though not all pieces of equipment are equipped with sensors yet, usually most of them are already capable of logging warnings and alarms occurring during operation.
Turning this data, which is easy to collect, into meaningful information about the health state of machinery can have a disruptive impact on the improvement of efficiency and up-time.

The provided dataset consists of a sequence of alarms logged by packaging equipment in an industrial environment. The collection includes data logged by 20 machines, deployed in different plants around the world, from 2019-02-21 to 2020-06-17. There are 154 distinct alarm codes, whose distribution is highly unbalanced.

This data can be used to address the following tasks:
1. Next alarm forecasting: this problem can be framed as a <b>supervised multi-class classification task</b>, or a binary classification task when a specific alarm code is considered.
2. Predicting alarms occurring in a future time frame: here the goal is to forecast the occurrence of certain alarm types in a future time window. Since many alarms can occur, this is a <b>supervised multi-label classification</b>.
3. Future alarm sequence prediction: here the goal is predicting an ordered sequence of future alarms, in a <b>sequence-to-sequence forecasting scenario</b>.
4. <b>Anomaly Detection</b>: the task is to detect abnormal equipment conditions, based on the pattern of alarms sequence. This task can be either unsupervised, if only the input sequence is considered, or supervised if future alarms are taken into account to assess whether or not there is an anomaly.

All of the above tasks can also be studied from a <b>continual learning perspective</b>. Indeed, information about the serial code of the specific piece of equipment can be used to train the model; however, a scalable model should also be easy to apply to new machines, without the need of a new training from scratch.


## Instructions
In this dataset, we provide both raw and processed data.
As for raw data, `raw/alarms.csv` is a comma-separated file with a row for each logged alarm. Each row provides the alarm code, the timestamp of occurrence, and the identifier of the piece of equipment generating the alarm. From this file, it is possible to generate data for tasks such as those described in the abstract.

For the sake of completeness, we also provide the Python code to process data and generate input and output sequences that can be used to address the task of predicting which alarms will occur in a future time window, given the sequence of all alarms occurred in a previous time window (`processed/all_alarms.pickle`, `processed/all_alarms.json`, and `processed/all_alarms.npz`).

The Python module to process raw data into input/output sequences is `dataset.py`. In particular, function `create_dataset` allows creating sequences already split in train/test and stored in a pickle file.
It is also possible to use `create_dataset_json` and `create_dataset_npz` to obtain different output formats for the processed dataset.

The ready-to-use datasets provided in the zipped folder were created by considering an input of 1720 minutes and an output window of 480 minutes.
More information can be found in the attached `readme.md` file.

## Detailed Instructions

### 1. Raw data
In `raw` folder you can find the comma-separated file `alarms.csv` which provides the list of all logged alarms.
It has the following schema:

| timestamp | alarm | serial |
|---|---|---|
|2019-02-21 19:57:57.532 | 139 |4 |

Therefore, for each alarm in this CSV file, the following details are provided: the timestamp, the type of alarm, and the machine that generated it (i.e. its identifier, also known as the serial number).

### 2. Processed data
In `processed` folder there is a processed dataset called `all_alarms` in three formats: <b>json, pickle, npz </b>. This dataset corresponds to input and output sequences that can be used to address the task of predicting which alarms will occur in a future time window, given the sequence of all alarms occurred in a previous time window.

The instructions to load the dataset based on the type of format are listed in the following subsections.

#### 2.1 Load data from pickle format
Pickle protocol used to save data requires python3:
```python
store_path = "processed/"
filename = "all_alarms.pickle"
filepath = os.path.join(store_path, filename)
with open(filepath, 'rb') as f:
    x_train, x_test, y_train, y_test, serials = pickle.load(f)

serials_train = serials['train']
serials_test = serials['test']

```

#### 2.2 Load data from json format
If you want load the dataset using the json format:
```python
from dataset import load_json_dataset
store_path = "processed"
filename_json= "all_alarms.json"
x_train,y_train,serials_train,x_test,y_test,serials_test = load_json_dataset(store_path, filename_json, verbose=1)
```

#### 2.3 Load data from npz format
If you want load the dataset using the npz format:
```python
from dataset import load_json_npz
store_path = "processed"
filename = "all_alarms.npz"
x_train, y_train, serials_train, x_test, y_test, serials_test = load_from_npz(store_path, filename, verbose=1)
```

### 3. Creation of dataset from raw data

The ready-to-use datasets provided in the zipped folder were created by considering the following parameters:
- `window_input` and `window_output` define the duration in minutes of input and output sequence, respectively.
- `offsets` defines the time shift (in minutes) applied to build partially overlapping input sequences.
- `sigma` and `min_count` are applied to constraint the length and informative content of input sequences.
- `removal_alarms`(optional) is a list of alarms to be disregarded in the definition of input sequences (if not defined, all alarms are kept)
- `relevance_alarms`(optional) is a list that can be used to keep only some of the alarms in output (all of them are kept if the list is empty or the value is not defined).

Dataset in `processed` called 'all_alarms' is created with the following instructions:

```python

ROOT = './'

DATA_PATH = ROOT + 'processed'

PARAMS = {'windows_input' : [1720],
          'windows_output' : [480],
          'min_counts' : [20],
          'sigmas' : [3],
          'offsets' : [60]}

RAW_DATA_FILE_PATH = f'raw/alarms.csv'

# load raw data
data = pd.read_csv(RAW_DATA_FILE_PATH, index_col=0, header=0, parse_dates=True)

 # create dataset params
params_list = create_params_list(DATA_PATH, PARAMS)

create_datasets(data, params_list)

```

#### 3.1 Formats of dataset

Dataset obtained with the previous code will be in pickle format.
To obtain other formats, like JSON and npz, additional functions are below.

##### 3.1.1 Dataset in json format

If you want convert dataset from pickle format (python3) to json format you can use the `convert_to_json` function:

```python
from dataset import convert_to_json
store_path = "processed" 
filename = "all_alarms.pickle"
convert_to_json(store_path, filename, verbose=1)
```

the structure of the JSON is the following:
```json
{
    ...

    "sample2001":{
            "x_train":[139, 97, 115, 139, 28, 11, 139, 20, 51, 139, 11, 0, 0, 0, ... ],
            "y_train":[11, 139, 11, 139, 11, 139, 11, 139, 11, 0, 0, 0, ...],
            "serial":3
            },

    ...

    "sample42001":{
        "x_test":[139, 30, 11, 139, 11, 139, 11, 97, 30, 97, 11, 139, 11, 139, 31, 139, 11, 0, 0, 0, ... ]
        "y_test":[139, 97, 11, 139, 19, 139, 0, 0, 0, ... ],
        "serial":3
        }

    ...
    

}
```

##### 3.1.2 Dataset in npz format

If you want convert dataset from pickle format to npz format(numpy) you can use the `convert_to_npz` function:

```python
from dataset import convert_to_npz
store_path = "processed"
filename = "all_alarms.pickle"
convert_to_npz(store_path, filename, verbose=1)
```

#### 4. Data quality assessment

When you load the dataset you should have six variables: `x_train, y_train, serials_train, x_test, y_test, serials_test`.
You can print their shape in the following way:
```python
print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)
print("serials_train.shape: ", np.asarray(serials_train).shape)
print("x_test.shape: ", x_test.shape)
print("y_test.shape: ", y_test.shape)
print("serials_test.shape: ", np.asarray(serials_test).shape)
```
The expected output is the following:
```python
x_train.shape:  (41045, 109)
y_train.shape:  (41045, 109)
serials_train.shape:  (41045,)
x_test.shape:  (17955, 109)
y_test.shape:  (17955, 109)
serials_test.shape:  (17955,)
```
