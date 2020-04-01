import numpy as np
import pandas as pd
import string
import itertools
import math


class Plate():
    def __init__(self, r, c, subwell_num):
        """
        General purpose plate lookup table. Converts a single number (Index) into a A01_1 format.
        Index is ordered left-right

        @param r: amount of rows
        @param c: amount of columns
        @param subwell_num: amount of subwells
        """
        self.row_num = r
        self.col_num = c
        self.subwell_num = subwell_num
        self.leading_zeros = int(math.log10(round(self.col_num, 0))) + 1
        self.zero_format = "{:0" + str(self.leading_zeros) + "d}"
        self.matrix = self.create_plate_matrix()

    def __str__(self):
        """
        Prints full table Pandas DataFrame as string
        @return: Pandas DataFrame as string
        """
        return str(self.matrix)


    def get_number_to_well_id(self, index):
        """
        Get well id (e.g. A01_1) from an Index (e.g. 1)
        @param index: Index (counting left-right what number well is it)
        @return: well_id
        """
        well_location = self.matrix[self.matrix == index].dropna(axis='index', how='all').dropna(axis='columns',
                                                                                                 how='all')
        if well_location.size > 0:
            col, subcol = well_location.columns.values[0]
            row = well_location.index.values[0]
        else:
            raise LookupError("%d doesn't exist in the plate" % index)
        return row+self.zero_format.format(int(col))+"_"+str(subcol)

    def create_plate_matrix(self):
        """
        Creates Pandas DataFrame of a plate for looking up well_ids
        @return: plate matrix
        """
        cols = list(str(n) for n in list(range(1, self.col_num + 1)))
        if self.row_num > len(string.ascii_letters):
            rows = itertools.product(string.ascii_uppercase, repeat=self.row_num // 27 + 1)
            rows = ["".join(pair) for pair in rows]
        else:
            rows = string.ascii_uppercase[:self.row_num]

        col_list = [[n] * self.subwell_num for n in cols]
        col_list_flat = []
        for l in col_list:
            for n in l:
                col_list_flat.append(n)

        multi_index_col_well_subwell_tuples = list(
            zip(col_list_flat, [i for _ in range(self.col_num) for i in range(1, self.subwell_num + 1)]))
        col_multindex = pd.MultiIndex.from_tuples(multi_index_col_well_subwell_tuples, names=['well', 'subwell'])
        well_ids = [[c + self.zero_format.format(int(n)) for n in cols] for c in rows]
        well_ids_flat = []
        i = 1
        for well in well_ids:
            well_subwell_one_letter = []
            for _ in well:
                for subwell in range(1, self.subwell_num + 1):
                    well_subwell_one_letter.append(i)
                    i += 1
            well_ids_flat.append(well_subwell_one_letter)

        return pd.DataFrame(well_ids_flat,
                            index=[c for c in rows],
                            columns=col_multindex)


