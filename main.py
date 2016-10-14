#!/usr/bin/python
# -*- coding: utf-8 -*-

from readers.mat_reader import MatReader


def main(mat_folder_path):
    mat_reader = MatReader(mat_folder_path)
    xs, ys = mat_reader.read()
    print '# of x = %d, # of y = %d' % (len(xs), len(ys))


# Simple example on how to read the data.
if __name__ == "__main__":
    main('ColdComplaintData/Training')
