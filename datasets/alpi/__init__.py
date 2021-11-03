import os
import shutil
import sys
import json
import pandas as pd
import numpy as np
import datetime
import bisect
import pickle
import time
from scipy import sparse
from sklearn.model_selection import train_test_split

FAMILY = 'MACHINE_TYPE_00'


# UTILS
def zip_dir(dir_path, zip_path):
    shutil.make_archive(zip_path, 'zip', dir_path)


def drop_files(dir_path, extension):
    test = os.listdir(dir_path)
    for item in test:
        if item.endswith(extension):
            os.remove(os.path.join(dir_path, item))


def prune(alarms):
    return alarms.loc[alarms.alarm.shift() != alarms.alarm].alarm.values.astype('uint16')


def prune_series(seq):
    series = pd.Series(seq)
    series = series[series.shift() != series].values
    return np.asarray(series)


def padding_sequence(seq, sequence_length):
    new_seq = np.zeros(sequence_length)
    new_seq[sequence_length - len(seq):] = seq
    return new_seq


def return_variables(params):
    window_input = params['window_input']
    window_output = params['window_output']
    offset = params['offset']
    verbose = params['verbose']
    store_path = params['store_path']  # store_path is used to save data
    min_count = params['min_count']
    sigma = params['sigma']
    return window_input, window_output, offset, verbose, store_path, min_count, sigma


def find_serials_offsets(store_path):
    filepath = store_path + "/" + store_path.split("/")[-1] + ".config"
    with open(filepath, 'rb') as f:
        serials, offsets = pickle.load(f)
    return serials, offsets


def return_index_output(date_range, target):
    # 0:start_output
    # 1:end_output
    # sample should be between start and end i.e. between i-1 and i where i must be odd
    i = bisect.bisect_right(date_range, target)
    if (i % 2) == 1:
        return int(i / 2)
    else:
        return -1  # element shouldn't be in any output


def return_index(date_range, target):
    # bisect: #return i such that i-1<timesamp<i
    i = bisect.bisect_right(date_range, target)
    return i - 1


def create_params_list(data_path, params, verbose=True):
    windows_input = params['windows_input']
    windows_output = params['windows_output']
    min_counts = params['min_counts']
    sigmas = params['sigmas']
    offsets = params['offsets']
    params_list = []
    dir_template = data_path + '/' + FAMILY + \
        '_alarms_window_input_{window_input}_window_output_{window_output}_offset_{offset}_min_count_{min_count}_sigma_{sigma}'
    for window_input in windows_input:
        for window_output in windows_output:
            for min_count in min_counts:
                for sigma in sigmas:
                    for offset in offsets:
                        store_path = dir_template.format(
                            window_input=window_input, window_output=window_output, offset=offset, min_count=min_count, sigma=sigma)
                        if not os.path.isdir(store_path):
                            os.makedirs(store_path)
                        params = {
                            'window_input': window_input,
                            'window_output': window_output,
                            'offset': offset,
                            'min_count': min_count,
                            'sigma': sigma,
                            'verbose': verbose,
                            'store_path': store_path
                        }
                        params_list.append(params)
    return params_list


# PHASE 1

# current_offset must be an integer and it must indicate minutes
def generate_dataset_by_serial_offset(data, params, current_offset):
    data["current_offset"] = current_offset
    current_offset = datetime.timedelta(minutes=current_offset)
    window_input, window_output, _, _, _, _, _ = return_variables(
        params)

    min_timestamp = data.index.min()
    min_timestamp += current_offset
    max_timestamp = data.index.max()

    # create date range for input
    date_range = pd.date_range(
        start=min_timestamp, end=max_timestamp, freq=str(window_input) + "min")
    date_range = [d for d in date_range]

    # create date range for output
    delta_in = datetime.timedelta(minutes=window_input)
    delta_out = datetime.timedelta(minutes=window_output)
    date_range_output = [(d + delta_in, d + delta_in + delta_out)
                         for d in date_range]  # ranges of outputs
    date_range_output = np.asarray(date_range_output).reshape(1, -1)[0]

    # create samples
    series = pd.Series(data.index).apply(
        lambda target: return_index(date_range, target))
    series.index = data.index
    data["bin_input"] = series

    series = pd.Series(data.index).apply(
        lambda target: return_index_output(date_range_output, target))
    series.index = data.index
    data["bin_output"] = series

    # to create a sample it is checked if for a bin input there is also a bin output and vice versa
    unique1 = data["bin_input"].unique()
    unique2 = data["bin_output"].unique()
    periods_id = list(set(unique1).intersection(set(unique2)) - set([-1]))

    data = data[(data["bin_input"].isin(periods_id)) |
                (data["bin_output"].isin(periods_id))]
    return data


def generate_dataset_by_serial(data, params):
    window_input, _, offset, verbose, store_path, _, _ = return_variables(
        params)
    serial = data["serial"][0]
    if verbose:
        print("serial: ", serial)
    data = data.sort_index()  # sort by index
    offsets = list(range(0, window_input, offset))
    offset_data = []
    for current_offset in offsets:
        df_offset = generate_dataset_by_serial_offset(data.copy(),
                                                      params,
                                                      current_offset)
        offset_data.append(df_offset)
    dataset = pd.concat(offset_data)
    if verbose:
        print("{} has shape: {}".format(str(serial) + ".csv", dataset.shape))
    filepath = os.path.join(store_path, str(serial) + ".csv")
    dataset.to_csv(filepath)


def generate_dataset(data, params):
    grouped_data = data.groupby('serial')
    for _, data_serial in grouped_data:
        generate_dataset_by_serial(data_serial, params)
    return


# PHASE 2

def prune_dataset(params):
    _, _, _, _, store_path, _, _ = return_variables(params)
    serials = []
    for f in os.listdir(store_path):
        if f.endswith(".csv"):
            serials.append(int(f.replace(".csv", "")))

    for serial in serials:
        df = pd.read_csv(os.path.join(store_path, str(serial) + ".csv"))
        offsets = prune_df(df, serial, params)
    return serials, offsets


def prune_df(df, serial_id, params):
    _, _, offset, verbose, store_path, min_count, _ = return_variables(
        params)

    offsets = df["current_offset"].unique()
    for offset in offsets:
        sub_df = df[df["current_offset"] == offset]
        if verbose:
            print("serial: {} offset: {}".format(serial_id, offset))

        groups_input = sub_df.groupby("bin_input")
        groups_output = sub_df.groupby("bin_output")

        unique1 = sub_df["bin_input"].unique()
        unique2 = sub_df["bin_output"].unique()
        periods_id = list(set(unique1).intersection(set(unique2)) - set([-1]))
        periods_id.sort()  # temporal sorting

        diz_input = {bin_id: group.alarm.sort_index(
        ).values for bin_id, group in groups_input if bin_id in periods_id}
        diz_output = {bin_id: group.alarm.sort_index(
        ).values for bin_id, group in groups_output if bin_id in periods_id}

        # list of [seq_x,seq_y]
        X_Y_offset = [[diz_input[bin_id], diz_output[bin_id]]
                      for bin_id in periods_id]

        # apply min_count: remove sequences with count<min_count
        X_Y_offset = [seq for seq in X_Y_offset if len(
            seq[0]) >= min_count and len(seq[1]) >= min_count]

        # save list on file
        # for each (serial,offset) a different list is saved
        if verbose:
            print("{} has length: {}".format(str(serial_id) +
                                             "_offset_" + str(offset) + ".npz", len(X_Y_offset)))
        filepath = store_path + "/" + \
            str(serial_id) + "_offset_" + str(offset) + ".npz"
        with open(filepath, 'wb') as f:
            pickle.dump(X_Y_offset, f)

    return offsets


# PHASE 3

def create_final_dataset(params, serials, offsets, sequence_input_length, sequence_output_length, removal_alarms=None, relevance_alarms=None, file_tag='all_alarms'):
    _, _, offset, verbose, store_path, _, sigma = return_variables(
        params)
    padding_mode = "after"
    if "padding_mode" in params:
        padding_mode = params["padding_mode"]
    X_train_tot = []
    X_test_tot = []
    Y_train_tot = []
    Y_test_tot = []
    list_X = []
    list_Y = []
    lengths_X = []
    lengths_Y = []
    cont = 0
    sentinel = 0
    tot_combo = len(serials) * len(offsets)
    combos = [(serial, offset) for serial in serials for offset in offsets]
    stratify = []
    for serial in serials:
        for offset in offsets:
            sentinel += 1
            if verbose:
                print("{}/{}:  serial: {}  offset: {}".format(sentinel,
                                                              tot_combo,
                                                              serial,
                                                              offset))
            filename = str(serial) + "_offset_" + str(offset) + ".npz"
            if filename in os.listdir(store_path):
                file_path = os.path.join(store_path, filename)
                with open(file_path, 'rb') as f:
                    seqences = pickle.load(f)
                    if removal_alarms != None and len(removal_alarms) > 0:
                        X = [[alarm for alarm in seq_x if (
                            alarm not in removal_alarms and alarm != 0)] for seq_x, seq_y in seqences]
                    else:
                        X = [seq_x for seq_x, seq_y in seqences]

                    if relevance_alarms != None and len(relevance_alarms) > 0:
                        Y = [[alarm for alarm in seq_y if (
                            alarm in relevance_alarms and alarm != 0)] for seq_x, seq_y in seqences]
                    else:
                        Y = [seq_y for seq_x, seq_y in seqences]

                    # pruning
                    X = [prune_series(seq_x) for seq_x in X]
                    Y = [prune_series(seq_y) for seq_y in Y]

                    list_X.append(X)
                    list_Y.append(Y)
                    lengths_X += [len(seq_x) for seq_x in X]
                    lengths_Y += [len(seq_y) for seq_y in Y]

    # length of sequences in input and output is calculated
    lengths_X = np.asarray(lengths_X)
    lengths_Y = np.asarray(lengths_Y)
    mu_sequence_input_length = lengths_X.mean()
    std_sequence_input_length = lengths_X.std()
    sequence_input_length = int(
        mu_sequence_input_length + sigma * std_sequence_input_length) + 1
    sequence_output_length = np.max(lengths_Y)
    sentinel = 0
    # it is iterated on all combinations of serials and offsets
    for i in range(len(list_X)):
        X = list_X[i]
        Y = list_Y[i]
        X = [seq_x[0:sequence_input_length] for seq_x in X]
        Y = [seq_y[0:sequence_output_length] for seq_y in Y]

        # PADDING
        if padding_mode == "after":
            X = [np.pad(seq_x, (0, sequence_input_length - len(seq_x)))
                 for seq_x in X]
            Y = [np.pad(seq_y, (0, sequence_output_length - len(seq_y)))
                 for seq_y in Y]
        elif padding_mode == "before":
            X = [padding_sequence(seq_x, sequence_input_length) for seq_x in X]
            Y = [padding_sequence(seq_y, sequence_output_length)
                 for seq_y in Y]

        if len(X) > 1:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.3, shuffle=False)
        else:
            X_train, X_test, Y_train, Y_test = [], [], [], []

        stratify.append({
            "serial": combos[sentinel][0],
            "offset": combos[sentinel][1],
            "X_train": len(X_train),
            "X_test": len(X_test),
            "Y_train": len(Y_train),
            "Y_test": len(Y_test)})

        cont += len(X_train) + len(X_test)
        X_train_tot += X_train
        X_test_tot += X_test
        Y_train_tot += Y_train
        Y_test_tot += Y_test
        sentinel += 1

    X_train = np.asarray(X_train_tot)
    X_test = np.asarray(X_test_tot)
    Y_train = np.asarray(Y_train_tot)
    Y_test = np.asarray(Y_test_tot)

    X_train = X_train.astype("int32")
    X_test = X_test.astype("int32")
    Y_train = Y_train.astype("int32")
    Y_test = Y_test.astype("int32")

    # for each sample we created the associated serial
    stratify_new = {}
    stratify_new["test"] = []
    stratify_new["train"] = []
    for diz in stratify:
        serial_test = [diz["serial"] for _ in range(diz["X_test"])]
        serial_train = [diz["serial"] for _ in range(diz["X_train"])]
        stratify_new["test"].extend(serial_test)
        stratify_new["train"].extend(serial_train)
    stratify = stratify_new

    # Sparse Matrix
    X_train, X_test, Y_train, Y_test = sparse.csr_matrix(X_train), sparse.csr_matrix(
        X_test), sparse.csr_matrix(Y_train), sparse.csr_matrix(Y_test)

    x_train_segments, x_test_segments, y_train_segments, y_test_segments = X_train, X_test, Y_train, Y_test

    if verbose:
        print(x_train_segments.shape)
        print(y_train_segments.shape)
        print(x_test_segments.shape)
        print(y_test_segments.shape)

    file_path = os.path.join(store_path, f"{file_tag}.pickle")
    with open(file_path, 'wb') as f:
        pickle.dump([x_train_segments, x_test_segments,
                     y_train_segments, y_test_segments, stratify], f)


def create_datasets(data, params_list, start_point=0, file_tag='all_alarms'):
    """
    Parameters:
        data : csv file of raw data
        params_list: list of dictionaries i.e. params 
                    each params is a dictionary with all parameters necessary to create the dataset
                    such as window_input, window_output, sigma, min_count,ecc.
        start_point: it tells from which phase to start to create dataset(i.e. if start_point=2 then it is assumed that phase1 is executed) 
        file_tag: name of dataset

    There are 3 phases(start_point:1,2,3):
    -PHASE1: it takes the csv  and create a file <serial>.csv for each serial
    -PHASE2: using files <serial>.csv for each (serial,offset) a file <serial>_offset_<offset>.npz is created.
             each file contains a list of x,y with pruning of consecutive alarms
    -PHASE3: some alarms in input are removed based on list associated with key 'removal_alarms' to variable params.
             only a subset of alarms in output is keeped based on a list associated with key 'relevance_alarms' to variable params.
             Padding is performed based on values defined by keys 'sequence_input_length' and 'sequence_output_length' in params  


    Example:
        windows_input=[480,960, 240]
        windows_output=[240]
        min_counts=[3]
        sigmas=[3]
        offsets=[230]
        verbose=True
        #several combinations of parameters are created, each combination can be considered as a new dataset
        params_list = create_params_list(windows_input, windows_output, min_counts, sigmas,  offsets, verbose)
        #removal_alarms = [] <- use it only if you want remove some alarms in input
        #relevance_alarms =  [] <- only if you want to keep a subset of alrams in output

        #additional parameters for phase3, comment the section if you don't want to use it
        for params in params_list:       
            params["removal_alarms"] = []
            params["relevance_alarms"] = []

        #create datasets
        create_datasets(params_list, start_point=0)
    """

    for params in params_list:
        print('-- run ', params)
        verbose = params["verbose"]
        if verbose:
            print(params)
        # phase 1
        start = time.time()
        if start_point <= 1:
            if verbose:
                print("phase 1")
            generate_dataset(data, params)
        end = time.time()
        elapsed_time = end - start
        print("phase 1/3 elapsed Time: {}".format(elapsed_time))

        # FASE2
        start = time.time()
        if start_point <= 2:
            if verbose:
                print("phase 2")
            serials, offsets = prune_dataset(params)
            # save serials and offsets
            store_path = params["store_path"]
            filepath = store_path + "/" + store_path.split("/")[-1] + ".config"
            with open(filepath, 'wb') as f:
                pickle.dump((serials, offsets), f)

        end = time.time()
        elapsed_time = end - start
        print("phase 2/3 elapsed Time: {}".format(elapsed_time))

        # phase 3
        start = time.time()
        if start_point <= 3:
            if verbose:
                print("phase 3")
                print()
            removal_alarms = None
            relevance_alarms = None
            sequence_input_length = None
            sequence_output_length = None
            serials, offsets = find_serials_offsets(params["store_path"])
            if verbose:
                print(serials)
                print(offsets)
            if "sequence_input_length" in params and "sequence_output_length" in params:
                sequence_input_length = params["sequence_input_length"]
                sequence_output_length = params["sequence_output_length"]
            if "removal_alarms" in params and "relevance_alarms" in params:
                removal_alarms = params["removal_alarms"]
                relevance_alarms = params["relevance_alarms"]
            create_final_dataset(params, serials, offsets, sequence_input_length,
                                 sequence_output_length, removal_alarms, relevance_alarms, file_tag)
            drop_files(params["store_path"], '.csv')
        end = time.time()
        elapsed_time = end - start
        print("phase 3/3 elapsed Time: {}".format(elapsed_time))


# POST BUILD

def clean(data_path):
    for filename_dir in os.listdir(data_path):
        print("directory: ", filename_dir)
        filepath_dir = data_path + '/' + filename_dir
        if os.path.isdir(filepath_dir):
            cont_csv, cont_npz, cont_pickle = 0, 0, 0
            for filename in os.listdir(filepath_dir):
                filepath = filepath_dir + "/" + filename
                if filename.endswith(".csv"):
                    cont_csv += os.stat(filepath).st_size
                elif filename.endswith(".npz"):
                    cont_npz += os.stat(filepath).st_size
                elif filename.endswith(".pickle"):
                    cont_pickle += os.stat(filepath).st_size
            print("cont_csv: {}MB  cont_npz: {}MB   cont_pickle: {}MB ".format(
                int(cont_csv / 1e6), int(cont_npz / 1e6), int(cont_pickle / 1e6)))
    for filename_dir in os.listdir(data_path):
        print("prune directory ", filename_dir)
        filepath_dir = data_path + '/' + filename_dir
        if os.path.isdir(filepath_dir):
            for filename in os.listdir(filepath_dir):
                filepath = filepath_dir + "/" + filename
                if filename.endswith(".csv"):
                    # print("delete file: ",filename)
                    os.remove(filepath)


# FUNCTIONS TO CONVERT PICKLE DATASET IN OTHER FORMATS(JSON, NPZ)


def convert_to_json(store_path, filename, verbose=0):
    """ 
    function to convert pickle dataset in json format which will have same name of original dataset
    but with json extension and stored in the same path of the original dataset

    Example:
            $#save data in json format
            $store_path = "../data/MACHINE_TYPE_00_alarms_window_input_1720_window_output_480_offset_60_min_count_20_sigma_3"
            $filename= "all_alarms.pickle"
            $convert_to_json(store_path, filename)

    Args:
        store_path (str): path that contains the dataset
        filename (str): name of dataset
    """
    # load dataset
    filepath = os.path.join(store_path, filename)
    with open(filepath, 'rb') as f:
        x_train_segments, x_test_segments, y_train_segments, y_test_segments, stratify = pickle.load(
            f)

    x_train_segments = x_train_segments.toarray()
    x_test_segments = x_test_segments.toarray()
    y_train_segments = y_train_segments.toarray()
    y_test_segments = y_test_segments.toarray()

    x_train_segments = x_train_segments.astype("int32")
    x_test_segments = x_test_segments.astype("int32")
    y_train_segments = y_train_segments.astype("int32")
    y_test_segments = y_test_segments.astype("int32")

    x_train_segments = x_train_segments.tolist()
    x_test_segments = x_test_segments.tolist()
    y_train_segments = y_train_segments.tolist()
    y_test_segments = y_test_segments.tolist()

    if verbose:
        print("x_train.shape: ", len(x_train_segments))
        print("y_train.shape: ", len(y_train_segments))
        print("serials_train.shape: ", len(stratify["train"]))
        print("x_test.shape: ", len(x_test_segments))
        print("y_test.shape: ", len(y_test_segments))
        print("serials_test.shape: ", len(stratify["test"]))

    diz = {}
    for i, (seq_x, seq_y, serial_id) in enumerate(zip(x_train_segments, y_train_segments, stratify["train"]), start=1):
        key = "sample"+str(i)
        diz[key] = {}
        diz[key]["x_train"] = seq_x
        diz[key]["y_train"] = seq_y
        diz[key]["serial"] = serial_id

    for i, (seq_x, seq_y, serial_id) in enumerate(zip(x_test_segments, y_test_segments, stratify["test"]), start=i+1):
        key = "sample"+str(i)
        diz[key] = {}
        diz[key]["x_test"] = seq_x
        diz[key]["y_test"] = seq_y
        diz[key]["serial"] = serial_id

    text = json.dumps(diz)
    filename_json = os.path.splitext(filename)[0]+".json"
    filepath_json = os.path.join(store_path, filename_json)
    if verbose:
        print("filepath_json: ", filepath_json)
    with open(filepath_json, 'w') as f:
        f.write(text)


def load_json_dataset(store_path, filename_json, verbose=0):
    """ 
    function to load dataset from json format

    Example:
            $#load data in json format
            $store_path = "../data/MACHINE_TYPE_00_alarms_window_input_1720_window_output_480_offset_60_min_count_20_sigma_3"
            $filename_json = "all_alarms.json"
            $filepath = os.path.join(store_path, filename_json)    
            $x_train,y_train,serials_train,x_test,y_test,serials_test = load_json_dataset(store_path, filename_json, verbose=1)


    Args:
        store_path (str): path that contains the dataset
        filename_json (str): name of dataset

    Returns:
        x_train: contain samples of input for train, each sample is a sequence of alarms
        y_train: contain samples of output for train, each sample is a sequence of alarms
        serials_train: for each train sample is indicated the serial that produced that sample
        x_test: contain samples of input for test, each sample is a sequence of alarms
        y_test: contain samples of output for test, each sample is a sequence of alarms
        serials_test: for each test sample is indicated the serial that produced that sample
    """

    filepath = os.path.join(store_path, filename_json)
    with open(filepath, 'r') as f:
        dataset = json.load(f)

    x_train = [sample["x_train"]
               for sample_id, sample in dataset.items() if "x_train" in sample]
    y_train = [sample["x_train"]
               for sample_id, sample in dataset.items() if "x_train" in sample]
    stratify_train = [sample["serial"] for sample_id,
                      sample in dataset.items() if "x_train" in sample]

    x_test = [sample["x_test"]
              for sample_id, sample in dataset.items() if "x_test" in sample]
    y_test = [sample["x_test"]
              for sample_id, sample in dataset.items() if "x_test" in sample]
    stratify_test = [sample["serial"]
                     for sample_id, sample in dataset.items() if "x_test" in sample]

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    if verbose:
        print("x_train.shape: ", x_train.shape)
        print("y_train.shape: ", y_train.shape)
        print("serials_train.shape: ", len(stratify_train))
        print("x_test.shape: ", x_test.shape)
        print("y_test.shape: ", y_test.shape)
        print("serials_test.shape: ", len(stratify_test))

    return x_train, y_train, stratify_train, x_test, y_test, stratify_test


def convert_to_npz(store_path, filename, verbose=0):
    """ 
    function to convert pickle dataset in npz format which will have same name of original dataset
    but with npz extension and stored in the same path of the original dataset

    Example:
            $#save data in npz format
            $store_path = "../data/MACHINE_TYPE_00_alarms_window_input_1720_window_output_480_offset_60_min_count_20_sigma_3"
            $filename= "all_alarms.pickle"    
            $convert_to_npz(store_path, filename, verbose=1)

    Args:
        store_path (str): path that contains the dataset
        filename (str): name of dataset
    """
    # load dataset
    filepath = os.path.join(store_path, filename)
    with open(filepath, 'rb') as f:
        x_train_segments, x_test_segments, y_train_segments, y_test_segments, stratify = pickle.load(
            f)

    x_train_segments = x_train_segments.toarray()
    x_test_segments = x_test_segments.toarray()
    y_train_segments = y_train_segments.toarray()
    y_test_segments = y_test_segments.toarray()

    x_train_segments = x_train_segments.astype("int32")
    y_train_segments = y_train_segments.astype("int32")
    stratify_train = np.asarray(stratify["train"])

    x_test_segments = x_test_segments.astype("int32")
    y_test_segments = y_test_segments.astype("int32")
    stratify_test = np.asarray(stratify["test"])

    if verbose:
        print("x_train.shape: ", x_train_segments.shape)
        print("y_train.shape: ", y_train_segments.shape)
        print("serials_train.shape: ", len(stratify["train"]))
        print("x_test.shape: ", x_test_segments.shape)
        print("y_test.shape: ", y_test_segments.shape)
        print("serials_test.shape: ", len(stratify["test"]))

    filename_npz = os.path.splitext(filename)[0]+".npz"
    filepath_npz = os.path.join(store_path, filename_npz)
    if verbose:
        print("filepath_npz: ", filepath_npz)
    np.savez_compressed(filepath_npz,
                        x_train=x_train_segments, y_train=y_train_segments, stratify_train=stratify_train,
                        x_test=x_test_segments, y_test=y_test_segments, stratify_test=stratify_test)


def load_from_npz():
    """ 
    function to load dataset from json format

    Example:
            $#load data from npz format
            $filename = "all_alarms.npz"
            $x_train_segments, y_train_segments, serials_train, x_test_segments, y_test_segments, serials_test = load_from_npz(store_path, filename)

    Args:
        store_path (str): path that contains the dataset
        filename (str): name of dataset in npz form

    Returns:
        x_train: contain samples of input for train, each sample is a sequence of alarms
        y_train: contain samples of output for train, each sample is a sequence of alarms
        serials_train: for each train sample is indicated the serial that produced that sample
        x_test: contain samples of input for test, each sample is a sequence of alarms
        y_test: contain samples of output for test, each sample is a sequence of alarms
        serials_test: for each test sample is indicated the serial that produced that sample
    """
    # filepath = os.path.join(store_path, filename)
    filepath = os.path.dirname(__file__) + '/alarms_log_data/processed/all_alarms.npz'
    loaded = np.load(filepath, allow_pickle=True)
    keys = ["x_train", "y_train", "stratify_train",
            "x_test", "y_test", "stratify_test"]
    return [loaded[k] for k in keys]
