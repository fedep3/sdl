# -*- coding: utf-8 -*-

import scipy.io as sio
import os


class MatReader:

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
        xs = []
        ys = []
        for f in os.listdir(self._folder_path):
            if f.endswith('.mat'):
                mat_data = sio.loadmat(os.path.join(self._folder_path, f))
                xs += [row[0:9] for row in mat_data['Info'][0][0][0]]
                ys += [row[10] for row in mat_data['Info'][0][0][0]]
        return xs, ys

    @property
    def folder_path(self):
        return self._folder_path

    @folder_path.setter
    def folder_path(self, folder_path):
        if folder_path is None:
            raise ValueError('Path cannot be null')
        self._folder_path = folder_path
