# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:43:47 2023

@author: Arvin Jay
"""

import pandas as pd
import googlemaps, time, itertools
from datetime import datetime as dt
import json

class DistanceMatrixCalculator:
    def __init__(self, hub: str = 'makati_hub', 
                 config_file: (str, dict) = 'config.json'):
        """
        Initialize the DistanceMatrixCalculator.

        Args:
            hub : str
                The hub for which the distance matrix is being calculated.
            config_file : str, dict
                Path to the configuration file containing API key and hub details.
        """
        if isinstance(config_file, str):
            with open(config_file) as json_file:
                config_file = json.load(json_file)
        elif isinstance(config_file, dict):
            pass
        else:
            raise Exception('Unable to load config_file.')
        
        api_key = config_file['mapsApiKey']
        # hubs = [key for key in config_file.keys() if key.endswith("_hub")]
        self.mechanigo_start = {
            'time_str': 'Duty start',
            'fullname': 'Mechanigo hub',
            'pin_address': hub,
            'lat': config_file[hub]['lat'],
            'long': config_file[hub]['long'],
            'service_category': 'In hub',
            'appointment_id': '0',
            'time': '05:00',
            'time_end':'05:01'
            }
        
        self.mechanigo_end = {'time_str':'Duty end',
                        'fullname':'Mechanigo hub',
                        'pin_address': hub,
                        'lat': config_file[hub]['lat'],
                        'long': config_file[hub]['long'],
                        'service_category':'Return to hub',
                        'appointment_id':'0',
                        'time': '19:00',
                        'time_end':'19:01'
                        }
        
        self.gmaps = googlemaps.Client(key=api_key)
        self.dataframe = None
    
    def time_parser(self, value: str):
        """
        Parse the time from a string.

        Args:
            value (str): Time in string format.

        Returns:
            Optional[dtime]: Parsed time or None if parsing fails.
        """
        try:
            value = dt.strptime(value,"%H:%M").time()
        except:
            value = None
        return value
    
    
    def load_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Load appointment data from a CSV file and prepare it for processing.

        Args:
            csv_file (str): Path to the CSV file containing appointment data.
        """
        rapide_start = self.mechanigo_start
        # {
        #     'time_str': 'Duty start',
        #     'fullname': 'Rapide Makati',
        #     'pin_address': '1166 Chino Roces Avenue, Corner Estrella, Makati, 1203',
        #     'lat': 14.5640785,
        #     'long': 121.0113147,
        #     'service_category': 'Depart from hub',
        #     'appointment_id': '0',
        #     'time': '05:00'
        # }
        if type(data) == type(pd.DataFrame()):
            df = data
        else:
            df = pd.read_csv(data)
        df_ = pd.DataFrame([rapide_start])
        df = pd.concat([df_, df],ignore_index = True)
        df = df[['time','time_end', 'pin_address', 'lat', 'long', 'service_category', 'service_names', 'appointment_id']]
        df['time'] = df['time'].apply(self.time_parser)
        df['time_end'] = df['time_end'].apply(self.time_parser)
        df['coord'] = df[['lat', 'long']].apply(lambda row: (row['lat'], row['long']), axis=1)
        self.dataframe = df
        return df

    def extract_info(self, df: pd.DataFrame, 
                     appointment_time: dt, 
                     destination: bool = True, 
                     col_extract: str = 'coord') -> list:
        """
        Extract location information based on appointment time.

        Args:
            appointment_time (time): Time of the appointment.
            destination (bool): If True, extract destinations; otherwise, extract sources.
            col_extract (str): Column to extract ('coord' or 'appointment_id').

        Returns:
            list: List of locations or appointment IDs.
        """
        if destination:
            return df.loc[df['time'] == appointment_time, col_extract].tolist()
        else:
            return df.loc[df['time_end'] <= appointment_time, col_extract].tolist()

    def gather_info(self, df: pd.DataFrame, 
                    appointment_time: dt) -> tuple[list, list]:
        """
        Gather source and destination information for a given appointment time.

        Args:
            appointment_time (time): Time of the appointment.

        Returns:
            tuple: Tuple of lists containing sources and destinations.
        """
        destinations = self.extract_info(df, appointment_time, True, 'coord')
        destinations_appids = self.extract_info(df, appointment_time, True, 'appointment_id')
        sources = self.extract_info(df, appointment_time, False, 'coord')
        sourcess_appids = self.extract_info(df, appointment_time, False, 'appointment_id')
        destination = list(zip(destinations, destinations_appids))
        source = list(zip(sources, sourcess_appids))
        return source, destination

    def split_list(self, locs: list, split_size: int) -> list[list]:
        """
        Split a list into sublists of a given size.

        Args:
            locs (List): List to split.
            split_size (int): Size of each sublist.

        Returns:
            List[List]: List of sublists.
        """
        locs_list = [locs[i:i + split_size] for i in range(0, len(locs), split_size)]
        return locs_list

    def split_source_destination(self, destinations: list, sources: list, split_size: int = 10) -> tuple[list, list]:
        """
        Split source and destination lists into sublists.

        Args:
            destinations (List): List of destinations.
            sources (List): List of sources.
            split_size (int): Size of each sublist.

        Returns:
            Tuple[List, List]: Tuple of destination and source sublists.
        """
        destinations = self.split_list(destinations, split_size)
        sources = self.split_list(sources, split_size)
        return destinations, sources

    def query_result(self, sources: list, 
                     destinations: list, 
                     arrival_time: dt, 
                     mode: str = 'driving') -> dict:
        """
        Query the Google Maps Distance Matrix API for travel information.

        Args:
            sources (List): List of source locations.
            destinations (List): List of destination locations.
            arrival_time (datetime): Arrival time for the travel.
            mode (str): Mode of transportation.

        Returns:
            dict: Results from the API.
        """
        results = self.gmaps.distance_matrix(sources, destinations, mode=mode, arrival_time=arrival_time)
        return results

    def construct_dist_time_df(self, results: dict, destinations_appids: list, sources_appids: list) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Construct distance and time DataFrames from API results.

        Args:
            results (dict): Results from the API.
            destinations_appids (List): List of destination appointment IDs.
            sources_appids (List): List of source appointment IDs.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Distance and time DataFrames.
        """
        distance_dict = {}
        time_dict = {}
        for i, destination in enumerate(destinations_appids):
            element_values_distance = [row["elements"][i]["distance"]["value"] for row in results["rows"]]
            element_values_time = [row["elements"][i]["duration"]["value"] for row in results["rows"]]
            distance_dict[destination] = element_values_distance
            time_dict[destination] = element_values_time
        distance_df = pd.DataFrame(distance_dict)
        time_df = pd.DataFrame(time_dict)
        distance_df.index = sources_appids
        time_df.index = sources_appids
        return distance_df, time_df

    def process_locations(self, sources: list, destinations: list) -> list:
        """
        Process source and destination lists into a list of combinations.

        Args:
            sources (List): List of source locations.
            destinations (List): List of destination locations.

        Returns:
            List: List of combinations.
        """
        query_list = list()
        combinations = itertools.product(sources, destinations)
        for combination in combinations:
            query_list.append(list(combination))
        return query_list

    def extract_coords_ids(self, tuple_list: list[tuple]) -> tuple[list, list]:
        """
        Extract coordinates and appointment IDs from a list of tuples.

        Args:
            tuple_list (List[Tuple]): List of tuples.

        Returns:
            Tuple[List, List]: Tuple of coordinates and appointment IDs.
        """
        coords = list()
        ids = list()
        for item in tuple_list:
            coords.append(item[0])
            ids.append(item[1])
        return coords, ids

    def query_timeslot(self, 
                       appointment_time: dt, 
                       my_date: dt, 
                       mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Query the Google Maps Distance Matrix API for a specific timeslot.

        Args:
            appointment_time (dtime): Time of the appointment.
            my_date (datetime): Date of the appointment.
            mode (str): Mode of transportation.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Distance and time DataFrames.
        """
        df_temp = self.dataframe.loc[self.dataframe['time'] <= appointment_time]
        arrival_time = dt.combine(my_date, appointment_time)
        source, destination = self.gather_info(df_temp, appointment_time)
        sources, destinations = self.split_source_destination(source, destination)
        query_list = self.process_locations(sources, destinations)
        distance_df_list = list()
        time_df_list = list()
        for query in query_list:
            source_locs, source_ids = self.extract_coords_ids(query[0])
            destination_locs, destination_ids = self.extract_coords_ids(query[1])
            results = self.query_result(source_locs, destination_locs, arrival_time, mode)
            distance_df, time_df = self.construct_dist_time_df(results, destination_ids, source_ids)
            distance_df_list.append(distance_df)
            time_df_list.append(time_df)
            time.sleep(1)
        try:
            distance_matrix_df = pd.concat(distance_df_list)
            time_matrix_df = pd.concat(time_df_list)
        except ValueError:
            distance_matrix_df = pd.DataFrame()
            time_matrix_df = pd.DataFrame()
        return distance_matrix_df, time_matrix_df

    def login_client(self, api_key: str = None) -> None:
        """
        Login to the Google Maps API.

        Args:
            api_key (str): Google Maps API key.
        """
        with open(r'.\lica_datascience_credentials.json') as creds:
            credentials = json.load(creds)
        
        self.gmaps = googlemaps.Client(key=credentials['gmapsApiKey'])

    def merge_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge two DataFrames.

        Args:
            df1 (pd.DataFrame): First DataFrame.
            df2 (pd.DataFrame): Second DataFrame.

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        return pd.merge(df1, df2, how='outer', left_index=True, right_index=True)

    def rewrite_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Rewrite the matrix by renaming and grouping.

        Args:
            matrix (pd.DataFrame): Input matrix.

        Returns:
            pd.DataFrame: Rewritten matrix.
        """
        matrix = matrix.rename_axis('index').reset_index()
        matrix = matrix.groupby('index').max()
        return matrix
    
    def process_df(self, current_date: dt, 
                   mode: str = 'driving') -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the entire DataFrame to get merged distance and time matrices.

        Args:
            current_date (datetime): Current date.
            mode (str): Mode of transportation.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Merged distance and time matrices.
        """
        partial_distance_matrix = list()
        partial_time_matrix = list()
        for appointment_time in self.dataframe.loc[self.dataframe['appointment_id'] != '0', 'time'].unique():
            if isinstance(appointment_time, str):
                appointment_time = pd.to_datetime(appointment_time).time()
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date).date()
            
            distance_matrix_df, time_matrix_df = self.query_timeslot(appointment_time, current_date, mode)
            distance_matrix_df = self.rewrite_matrix(distance_matrix_df)
            time_matrix_df = self.rewrite_matrix(time_matrix_df)
            if len(distance_matrix_df)>0:
                partial_distance_matrix.append(distance_matrix_df)
                partial_time_matrix.append(time_matrix_df)
        merged_distance_matrix = pd.concat(partial_distance_matrix,axis=1)
        merged_time_matrix = pd.concat(partial_time_matrix,axis=1)
        return merged_distance_matrix, merged_time_matrix
    
    
    
if __name__ == "__main__":
    
    calculator = DistanceMatrixCalculator()
    dataframe = calculator.load_data('appointment_temp.csv')
    current_date = dt(2023, 10, 18).date()
    partial_distance_matrix = list()
    partial_time_matrix = list()
    for appointment_time in dataframe.loc[dataframe['appointment_id'] != '0', 'time'].unique():
        distance_matrix_df, time_matrix_df = calculator.query_timeslot(appointment_time, current_date, 'driving')
        partial_distance_matrix.append(distance_matrix_df)
        partial_time_matrix.append(time_matrix_df)
    merged_distance_matrix = pd.concat(partial_distance_matrix,ignore_index=True, axis=1)
    merged_time_matrix = pd.concat(partial_time_matrix,ignore_index=True, axis=1)
    # You can now work with the merged distance and time matrices.
