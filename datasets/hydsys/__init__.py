import os
import numpy as np
import pandas as pd
import tqdm


def load_data(sensor=None, rw=0):
    
    if sensor is None:
        # load full data
        # rw is ignored to concatenate all sensor data
        df = []
        df.append(load_sensor_data('PS', rw=10))  # default length: 6000
        df.append(load_sensor_data('EPS', rw=10))  # default length: 6000
        df.append(load_sensor_data('FS', rw=0))  # default length: 600
        # df.append(load_sensor_data('TS', rw=0))  # default length: 60
        return pd.concat(df)

    else:
        return load_sensor_data(sensor, rw)
        

def load_sensor_data(sensor, rw=0):

    data = []
    sensor_list = get_sensor_list(sensor)
    fp = os.path.dirname(__file__)

    for name in tqdm.tqdm(sensor_list, desc=sensor):
        df = pd.DataFrame(np.loadtxt(fp + f'/{name}.txt.gz'))
        df = resample(df, rw)
        df['sensor'] = name
        df['cycle'] = df.index.values
        data.append(df)

    return pd.concat(data).set_index(['cycle', 'sensor']).reset_index()


def get_sensor_list(name):
    if name == 'PS':
        return [f'PS{i+1}' for i in range(6)]
    elif name == 'EPS':
        return ['EPS1']
    elif name == 'FS':
        return [f'FS{i+1}' for i in range(2)]
    elif name == 'TS':
        return [f'TS{i+1}' for i in range(4)]
    elif name == 'VS':
        return ['VS1']
    else:
        raise ValueError


def load_labels():
    fp = os.path.dirname(__file__)
    return pd.DataFrame(np.loadtxt(fp + '/profile.txt'),
        columns=[
            'cooler_condition',
            'valve_condition',
            'internal_pump_leakage',
            'hydraulic_accumulator',
            'stable_flag']).reset_index().rename(
                columns={'index': 'cycle'})


def resample(df, rw=0):
    if rw > 0:
        # Resampling
        df = df.T
        df['index'] = df.index.values // rw
        df = df.groupby('index').mean()
        df = df.T

    return df


