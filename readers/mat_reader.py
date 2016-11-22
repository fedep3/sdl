# -*- coding: utf-8 -*-

import scipy.io as sio
import os
from datetime import datetime, timedelta


class MatReader:

    BEGINNING_DATETIME = datetime(2014, 11, 3, 0, 0, 0)

    def __init__(self):
        pass

    """
    It will read different .mat files in the folder_path and will merge the data.
    :return Two lists, the first one the Xs values and the second one the corresponding Ys values.
    """
    def read(self, folder_path):
        ts = []
        xs = []
        ys = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.mat'):
                n = int(filename.lstrip('data').rstrip('.mat')) - 1
                starting_datetime = self.BEGINNING_DATETIME + timedelta(days=n)
                mat_data = sio.loadmat(os.path.join(folder_path, filename))
                for row in mat_data['Info'][0][0][0]:
                    ts.append(starting_datetime)
                    xs.append(row[0:9])
                    ys.append(row[10])
                    starting_datetime = starting_datetime + timedelta(minutes=5)
        return ts, xs, ys
