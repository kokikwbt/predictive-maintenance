import os
import pandas as pd
import numpy as np


def load_data(meter_id='A'):
    fp = os.path.dirname(__file__)
    data = np.loadtxt(fp + '/Meter{}.txt'.format(meter_id))

    if meter_id == 'A':
        columns =  ['flatness_ratio', 'symmetry', 'crossflow']
        columns += ['flow_velocity_{}'.format(i+1) for i in range(8)]
        columns += ['sound_speed_{}'.format(i+1) for i in range(8)]
        columns += ['average_speed']
        columns += ['gain_{}'.format(i+1) for i in range(16)]
        columns += ['health_state']

    if meter_id == 'B':
        columns =  ['profile_factor', 'symmetry', 'crossflow', 'swirl_angle']
        columns += ['flow_velocity_{}'.format(i+1) for i in range(4)]
        columns += ['average_flow']
        columns += ['sound_speed_{}'.format(i+1) for i in range(4)]
        columns += ['average_speed']
        columns += ['signal_strength_{}'.format(i+1) for i in range(8)]
        columns += ['turbulence_{}'.format(i+1) for i in range(4)]
        columns += ['meter_performance']
        columns += ['signal_quality_{}'.format(i+1) for i in range(8)]
        columns += ['gain_{}'.format(i+1) for i in range(8)]
        columns += ['transit_time_{}'.format(i+1) for i in range(8)]
        columns += ['health_state']

    if meter_id == 'C' or meter_id == 'D':
        columns =  ['profile_factor', 'symmetry', 'crossflow']
        columns += ['flow_velocity_{}'.format(i+1) for i in range(4)]
        columns += ['sound_speed_{}'.format(i+1) for i in range(4)]
        columns += ['signal_strength_{}'.format(i+1) for i in range(8)]
        columns += ['signal_quality_{}'.format(i+1) for i in range(8)]
        columns += ['gain_{}'.format(i+1) for i in range(8)]
        columns += ['transit_time_{}'.format(i+1) for i in range(8)]
        columns += ['health_state']

    return pd.DataFrame(data, columns=columns)