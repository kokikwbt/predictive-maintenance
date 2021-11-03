import os
import numpy as np
import pandas as pd


def make_pressure_dataframe(fp, cycle_id=0):
    """ 100 Hz. 6000 samples in each cycle
    """
    data = pd.DataFrame(columns=[f'PS{i+1}' for i in range(6)])
    for i in range(6):
        data[f'PS{i+1}'] = np.loadtxt(fp + f'/PS{i+1}.txt.gz')[cycle_id]

    return data


def make_motor_power_dataframe(fp, cycle_id=0):
    """ 100 Hz. 6000 samples in each cycle
    """
    return pd.DataFrame(np.loadtxt(fp + '/EPS1.txt.gz')[cycle_id], columns=['EPS1'])


def make_volume_flow_dataframe(fp, cycle_id=0):
    """ 10 Hz. 600 samples in each cycle
    """
    data = pd.DataFrame(columns=['FS1', 'FS2'])
    data['FS1'] = np.loadtxt(fp + '/FS1.txt.gz')[cycle_id]
    data['FS2'] = np.loadtxt(fp + '/FS2.txt.gz')[cycle_id]
    return data


def make_temp_dataframe(cycle_id=0):
    """ 1 Hz. 60 samples in each cycle
    """
    fp = os.path.dirname(__file__)
    data = pd.DataFrame(columns=[f'TS{i+1}' for i in range(4)])
    data['TS1'] = np.loadtxt(fp + '/TS1.txt.gz')[cycle_id]
    data['TS2'] = np.loadtxt(fp + '/TS2.txt.gz')[cycle_id]
    data['TS3'] = np.loadtxt(fp + '/TS3.txt.gz')[cycle_id]
    data['TS4'] = np.loadtxt(fp + '/TS4.txt.gz')[cycle_id]
    return data


def make_vibration_dataframe(cycle_id=0):
    fp = os.path.dirname(__file__)
    return pd.DataFrame(np.loadtxt(fp + '/VS1.txt.gz')[cycle_id], columns=['VS1'])


def make_efficiency_dataframe(cycle_id=0):
    fp = os.path.dirname(__file__)
    return pd.DataFrame(np.loadtxt(fp + '/SE.txt.gz')[cycle_id], columns=['SE'])


def make_cooling_dataframe(cycle_id=0):
    fp = os.path.dirname(__file__)
    data = pd.DataFrame(columns=['CE', 'CP'])
    data['CE'] = np.loadtxt(fp + '/CE.txt.gz')[cycle_id]
    data['CP'] = np.loadtxt(fp + '/CP.txt.gz')[cycle_id]
    return data


def make_condition_dataframe():
    fp = os.path.dirname(__file__)
    return pd.DataFrame(np.loadtxt(fp + '/profile.txt'),
        columns=[
            'cooler_condition',
            'valve_condition',
            'internal_pump_leakage',
            'hydraulic_accumulator',
            'stable_flag']).reset_index().rename(
                columns={'index': 'cycle'})


