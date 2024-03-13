# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:52:30 2023

@author: Arvin Jay
"""

import pandas as pd
import numpy as np

class matrixConditioning:
    def __init__(self, 
                 time_matrix_file='time_matrix.csv', 
                    distance_matrix_file='distance_matrix.csv',
                    appointments_file = 'sample_appointments.csv'):
        self.time_matrix_file = time_matrix_file
        self.distance_matrix_file = distance_matrix_file
        self.appointments_file = appointments_file
        self.time_matrix = None
        self.distance_matrix = None
        self.appointments = None
        self.inf_number = 9999999
        
    def gather_appointments(self):
        """
        Gathers appointment data from the specified CSV file.

        Returns:
            pd.DataFrame: Appointment data.
        """
        if isinstance(self.appointments_file,str):
            self.appointments = pd.read_csv(self.appointments_file)
        else:
            self.appointments = self.appointments_file
        return self.appointments
    
    def process_matrix(self, file):
        """
        Processes the matrix from the specified CSV file.

        Args:
            file (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Processed matrix.
        """
        df_matrix = pd.read_csv(file)
        df_matrix = df_matrix.rename(columns={'Unnamed: 0': 'index'})
        df_matrix = df_matrix.set_index('index')
        df_matrix.index = [ str(i) for i in df_matrix.index.tolist()]
        df_matrix.columns = [ str(i) for i in df_matrix.columns.tolist()]
        return df_matrix
        
    
    def gather_data(self):
        """
        Load time and distance matrices from CSV files and prepare them for processing.

        Returns:
            pd.DataFrame: Time matrix.
            pd.DataFrame: Distance matrix.
        """
        if type(self.time_matrix_file) == type(pd.DataFrame()):
            return self.time_matrix_file, self.distance_matrix_file
        else:
            # self.time_matrix = pd.read_csv(self.time_matrix_file)
            # self.distance_matrix = pd.read_csv(self.distance_matrix_file)
            # self.time_matrix = self.time_matrix.rename(columns={'Unnamed: 0': 'index'})
            # self.distance_matrix = self.distance_matrix.rename(columns={'Unnamed: 0': 'index'})
            # self.time_matrix = self.time_matrix.set_index('index')
            # self.distance_matrix = self.distance_matrix.set_index('index')
            # self.distance_matrix.index = [ str(i) for i in distance_matrix.index.tolist()]
            # self.distance_matrix.columns = [ str(i) for i in distance_matrix.columns.tolist()]
            self.time_matrix = self.process_matrix(self.time_matrix_file)
            self.distance_matrix =  self.process_matrix(self.distance_matrix_file)
        return self.time_matrix, self.distance_matrix

    def gather_index_columns(self, distance_matrix):
        """
        Get the index and column lists from the distance matrix.

        Args:
            distance_matrix (pd.DataFrame): Distance matrix.

        Returns:
            list: List of indices.
            list: List of columns.
        """
        index_list = distance_matrix.index.tolist()
        column_list = distance_matrix.columns.tolist()
        return index_list, column_list

    def create_empty_df(self, column_list, add_to_index):
        """
        Create an empty DataFrame with specified columns and index.

        Args:
            column_list (list): List of columns.
            add_to_index (list): List of indices to add.

        Returns:
            pd.DataFrame: Empty DataFrame.
        """
        result = {}
        for column in column_list:
            result[column] = [None] * len(add_to_index)
        return pd.DataFrame(result, index=add_to_index)

    def add_indices(self, distance_matrix):
        """
        Add missing indices to the distance matrix.

        Args:
            distance_matrix (pd.DataFrame): Distance matrix.

        Returns:
            pd.DataFrame: Updated distance matrix.
        """
        index_list, column_list = self.gather_index_columns(distance_matrix)
        add_to_index = list(filter(lambda x: x not in index_list, column_list))
        empty_df = self.create_empty_df(column_list, add_to_index)
        # distance_matrix = distance_matrix.append(empty_df)
        distance_matrix = pd.concat([distance_matrix,empty_df])
        return distance_matrix

    def add_depot_column(self, distance_matrix):
        """
        Add a depot column to the distance matrix.

        Args:
            distance_matrix (pd.DataFrame): Distance matrix.

        Returns:
            pd.DataFrame: Updated distance matrix.
        """
        
        
        append_column = distance_matrix.T['0']
        distance_matrix.insert(0, '0', append_column)
        return distance_matrix

    def zero_diagonal(self, distance_matrix):
        """
        Set the diagonal elements of the distance matrix to zero.

        Args:
            distance_matrix (pd.DataFrame): Distance matrix.

        Returns:
            pd.DataFrame: Updated distance matrix.
        """
        array = distance_matrix.values
        np.fill_diagonal(array, 0)
        distance_matrix = pd.DataFrame(array, columns=distance_matrix.columns, index=distance_matrix.index)
        return distance_matrix
    
    def fill_inf(self, distance_matrix):
        """
        Fills NaN values in the distance matrix with a large number.

        Args:
            distance_matrix (pd.DataFrame): Distance matrix.

        Returns:
            pd.DataFrame: Updated distance matrix.
        """
        distance_matrix = distance_matrix.fillna(self.inf_number)
        return distance_matrix
    
    def main_conditioning(self,matrix):
        """
        Perform main conditioning operations on the distance matrix.

        Returns:
            pd.DataFrame: Conditioned distance matrix.
        """
        matrix = self.add_indices(matrix)
        matrix = self.add_depot_column(matrix)
        matrix = self.zero_diagonal(matrix)
        matrix = self.fill_inf(matrix)
        return matrix

if __name__ == "__main__":
    processor = matrixConditioning(time_matrix_file='sample_mdm.csv', 
                                    distance_matrix_file='sample_dm.csv',
                                    appointments_file = 'appointments_sample.csv')
    appointments = processor.gather_appointments()
    time_matrix, distance_matrix = processor.gather_data()
    appointments = appointments[appointments['hub_solver'] == 'makati_hub']
    conditioned_distance_matrix = processor.main_conditioning(time_matrix)
    # You can now work with the conditioned distance matrix.
    
    
    