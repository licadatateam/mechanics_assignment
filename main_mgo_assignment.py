# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:32:13 2024

@author: carlo
"""

import pandas as pd
import json, re
from datetime import datetime as dt
from datetime import timedelta
from fuzzywuzzy import fuzz, process

# custom modules
from barangay_processing import geocode_by_barangay
import orMechanicAssignment, clustering_methods
from gmaps_distance_matrix import DistanceMatrixCalculator
from matrix_conditioning import matrixConditioning

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(message)s',
                    datefmt = '%d-%b-%y %H:%M:%S',
                    level=logging.WARNING)

def load_config(config_file : str = 'config') -> dict:
    '''
    
    Load configuration data and secrets
    
    google sheet link:
    https://docs.google.com/spreadsheets/d/1C4XVFaYPCMPtRFE5X_sHO_a89QCmU2bVzmnMTWwrkN8/edit#gid=0
    
    Args:
    -----
        - config_file : str
            configuration filename (no extension)
    
    Returns:
    --------
        - keys : dict
            config file dict
    
    '''
    # load configuration and secrets
    config_file = f'{config_file}.json'
    with open(config_file) as json_file:
        keys = json.load(json_file)
    
    # TODO: convert to redash query
    barangays = {'sheet_id': keys['barangays_sheet_id'],
                 'tab': keys['barangays_tab']}
    orders = {'sheet_id': keys['orders_sheet_id'],
                 'tab': keys['orders_tab']}
    
    keys['barangays'] = barangays
    keys['orders'] = orders
    return keys

def extract_hub(entry: str):
    """
    Extract hub information from the given entry.

    Args:
    -----
        - entry : str 
            The entry to process.

    Returns:
    --------
        - res : str 
            The extracted hub information, or None if not found.
    """
    hubs = ['makati_hub', 'sucat_hub', 'starosa_hub']
    
    try:
        _ = process.extractOne(entry, hubs, 
                           scorer = fuzz.partial_ratio,
                           score_cutoff = 75)
    except:
        _ = None
    
    # if result satisfying matching cutoff is found
    if _ is not None:
        res = _[0]
    # if no match is found
    else:
        res = None
    
    return res

def gather_data(key: str, 
                selected_date: (dt, str)) -> pd.DataFrame:
    """
    Gather appointment data from the specified key and selected date.

    Args:
    ------
        - key : str
            The key to access appointment data (redashApiKey)
        - selected_date : date 
            The selected date for filtering appointments.

    Returns:
    --------
        - appointments : pd.DataFrame 
            The gathered cleaned appointment data.
    """
    
    try:
        appointments = pd.read_csv(key)
        logging.debug('Importing appointments data from redash.')
    except Exception as e:
        raise e
    
    try:
        appointments['date'] = pd.to_datetime(appointments['date'], format = '%m/%d/%y')
        appointments['time'] = pd.to_datetime(appointments['time'], format = '%H:%M:%S')
        # TODO: adjustable service duration
        appointments['time_end'] = appointments['time'].apply(lambda x: x+timedelta(hours=2, minutes=15)).dt.time
        appointments[['time','time_end']] = appointments[['time','time_end']].map(lambda x: x.strftime('%H:%M'))
        # TODO: clean province from redash
        appointments['province'] = appointments['province'].str.title()
        appointments['address'] = appointments['address'].fillna('')
        # filter appointments on date
        try:
            # assume type str, if date, trigger exception
            appointments = appointments.loc[appointments['date']==selected_date.strftime('%Y-%m-%d')].drop(columns = ['date'])
            #appointments = appointments.loc[appointments['date']==selected_date.strftime('%Y-%m-%d')]
        except:
            appointments = appointments.loc[appointments['date']==selected_date].drop(columns = ['date'])
            #appointments = appointments.loc[appointments['date']==selected_date]
        # drop duplicates
        appointments = appointments.drop_duplicates(subset = ['appointment_id']).reset_index(drop = True)
        
        # if hub is explicitly stated, else None meaning need to be solved
        appointments['hub_solver'] = appointments['pin_address'].apply(extract_hub)
        logging.debug('Cleaned appointments data.')
        
    except Exception as e:
        raise e
        
    return appointments

def clean_address(row: pd.Series, 
                  address_col: str = 'pin_address', 
                  barangay_col: str = 'barangay',
                  municipality_col: str = 'municipality', 
                  province_col: str = 'province') -> tuple[str, str, str, str, str]:
    """
    Clean and format the address information in a given row of a DataFrame.

    Args:
    -----
        - row : pd.Series
            The row containing address-related columns.
        - address_col : str (Optional)
            Column name for the address. Defaults to 'pin_address'.
        - barangay_col : str (Optional)
            Column name for the barangay. Defaults to 'barangay'.
        - municipality_col : str (Optional)
            Column name for the municipality. Defaults to 'municipality'.
        - province_col : str (Optional)
            Column name for the province. Defaults to 'province'.

    Returns:
        - Tuple[str, str, str, str, str]: A tuple containing the cleaned street address, street name, cleaned full address,
        full address with street name, and full address with street name and barangay.

    Examples:
        >>> import pandas as pd
        >>> data = {'pin_address': '123 Main St, Brgy. Example, Municipality A, Province X', 'barangay': 'Example', 'municipality': 'Municipality A', 'province': 'Province X'}
        >>> row = pd.Series(data)
        >>> clean_address(row)
        ('123 Main St', 'Main St', '123 Main St in Example, Municipality A, Province X in Philippines', 'Main St in Brgy. Example, Municipality A, Province X in Philippines', 'Main St in Example, Municipality A, Province X in Philippines')
    """
    address = row[address_col] if row[address_col] else ''
    barangay = row[barangay_col] if row[barangay_col] else ''
    municipality = row[municipality_col] if row[municipality_col] else ''
    province = row[province_col] if row[province_col] else ''

    # Additional logic to handle specific cases
    if province.lower() == 'metro manila' and municipality.lower() == 'quezon':
        municipality = 'quezon city'

    remove_pattern = re.compile(r'b\.?f\.? homes')

    if len(str(address)) != 0:
        address = re.sub(remove_pattern, '', str(address))
    
    try:
        if barangay in address:
            street_address = address.split(barangay)[0].strip()
        else:
            if municipality in address:
                street_address = address.split(municipality)[0].strip()
            else:
                if province in address:
                    street_address = address.split(municipality)[0].strip()
                else:
                    street_address = address
                    
        cleaned_address_ = street_address
        if street_address != address:
            cleaned_address = ' '.join([street_address, 'in', barangay + ',', municipality + ',', province, 'in Philippines'])
        else:
            cleaned_address = street_address + ' in Philippines'
    except:
        street_address = address
        cleaned_address_ = address
        cleaned_address = address

    street = ''
    # Street pattern
    st_pattern = r'(\b\w+\s+(st\.?|street)\b)'
    match = re.search(st_pattern, street_address, re.IGNORECASE)
    if match:
        street = match.group(1)
    cleaned_address_ = re.sub(street, '', cleaned_address_).strip()

    try:
        street_address_ = ' '.join([street, 'brgy.', barangay + ',', municipality + ',', province, 'in Philippines'])
    except:
        street_address_ = street_address

    return street_address, street, cleaned_address, cleaned_address_, street_address_


def clean_address_appts(appointments : pd.DataFrame) -> pd.DataFrame:
    '''
    Wrapper function for clean_address
    
    Args:
    -----
        - appointments : pd.DataFrame
            appointments data
    
    Returns:
    --------
        - _appointments : pd.DataFrame
            copy of appointments data with cleaned address (if no errors)
    '''
    _appointments = appointments.copy()
    
    try:
        _appointments[['street_address', 'street', 'address_query', 'partial_address',
                      'street_address_query']] = _appointments.apply(clean_address, 
                                                                    axis=1, 
                                                                    result_type='expand')
        logging.debug('Cleaning appointments addresses.')
    except Exception as e:
        logging.exception(e)
    
    return _appointments

def geocode(barangays: dict, 
            appointments: pd.DataFrame) -> dict:
    """
    Perform geocoding based on barangays and appointments data.

    Args:
    -----
        - barangays : dict
            Dictionary containing information about barangays, typically loaded from configuration.
        - appointments : pd.DataFrame
            DataFrame containing appointment data.

    Returns:
    --------
        - geo_dict : dict {pd.DataFrame, pd.DataFrame, Optional[str]}
            A dict containing geocoded appointments DataFrame, a review DataFrame,
        and an error message if any.
    """
    try:
        _ = geocode_by_barangay(barangays, appointments)
        geo_dict = {'appointments' : _[0],
                    'review' : _[1],
                    'error' : _[2]}
        
        logging.debug('Geocoding addresses.')
        return geo_dict
    
    except Exception as e:
        raise e

def main(selected_date : dt = dt.today().date(),
         mechanic_counts_dict : dict = None):
    
    
    # configuration settings
    config_dict = load_config()
    
    # Load Redash query result
    appointments = gather_data(config_dict['redashApiKey'], selected_date)
    
    # clean address
    appointments = clean_address_appts(appointments)

    # geocoding
    geo_dict = geocode(config_dict['barangays'], appointments)
    appointments = geo_dict['appointments']
    
    if mechanic_counts_dict is None:
        mechanic_counts_dict = {'makati_hub': 15,
                                'starosa_hub': 15,
                                'sucat_hub': 1}
    
    appointments,similarity,mechanics_per_hub,df_hub_service = clustering_methods.cluster_appointments(appointments,
                                                                                                       config_dict,
                                                                                                       mechanic_counts_dict)
    appointments.loc[:, 'appointment_date'] = dt.today().date().strftime('%Y-%m-%d')
    # appointments.loc[:, 'created_at'] = pd.NaT
    # appointments.loc[:, 'updated_at'] = pd.NaT
    # if appointments['created_at'].isnull().sum() == len(appointments):
    #     appointments.loc[:, 'created_at'] = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    # else:
    #     appointments.loc[:, 'updated_at'] = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    
    show_cols = ['appointment_date', 'appointment_id', 'hub_solver']
    
    return appointments[show_cols].to_json(orient = 'index')
    