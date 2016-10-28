# -*- coding: utf-8 -*-

import scipy.io as sio
import os
from datetime import datetime, timedelta


class MatReader:

    BEGINNING_DATETIME = datetime(2014, 11, 3, 0, 0, 0)

    """
    :folder_path Path to the folder containing the .mat files.
    """
    def __init__(self, folder_path):
        self._folder_path = folder_path

    """
    It will read different .mat files in the folder_path and will merge the data.
    :return Two lists, the first one the Xs values and the second one the corresponding Ys values.
    """
    def read(self):
        ts = []
        xs = []
        ys = []
        for filename in os.listdir(self._folder_path):
            if filename.endswith('.mat'):
                n = int(filename.lstrip('data').rstrip('.mat')) - 1
                starting_datetime = self.BEGINNING_DATETIME + timedelta(days=n)
                mat_data = sio.loadmat(os.path.join(self._folder_path, filename))
                for row in mat_data['Info'][0][0][0]:
                    ts.append(starting_datetime)
                    xs.append(row[0:9])
                    ys.append(row[10])
                    starting_datetime = starting_datetime + timedelta(minutes=5)
        return ts, xs, ys

    @property
    def folder_path(self):
        return self._folder_path

    @folder_path.setter
    def folder_path(self, folder_path):
        if folder_path is None:
            raise ValueError('Path cannot be null')
        self._folder_path = folder_path
