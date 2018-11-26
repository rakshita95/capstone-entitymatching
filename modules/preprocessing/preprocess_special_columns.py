"""



"""

import re
import numpy as np
import pandas as pd
from geopy.geocoders import GoogleV3
from .process_text import Process_text

def geocode_address_api(x, api_key=None):
    """
    takes a list of values of address fields, and geocodes it using the goolge
    geocoding API.
    :param x: str | the address string or a list of strings
    :param api_key: str | the api_key
    :return:
    """
    if not(api_key):
        print("Geocoding Failed! Please provide an API key")
        return None, None, None

    if type(x) == list or type(x) == np.ndarray or type(x) == pd.core.series.Series:
        address = ', '.join(x)
    else:
        address = x
    api = GoogleV3(api_key=api_key)
    loc = api.geocode(address)
    if loc:
        return loc.address, loc.latitude, loc.longitude
    else:
        return None, None, None

def preprocess_phone(number):
    """

    :param number:
    :return:
    """
    number = str(number)
    number = number.replace('+','00')
    number = re.sub(r'\W+', '', number)

    return number

def preprocess_email(mail):
    """

    :param mail:
    :return:
    """
    # TODO: find a way to normalize e-mail https://pypi.org/project/email-normalize/
    return None


def preprocess_special_fields(df,
                              phone_number=None,
                              address_columns=None,
                              geocode_address=False,
                              api_key=None):
    """
    This function preprocesses the special fields columns and uses the google
    geocode API to return a normalized address as well as longitude and latitutde data.
    If the google API cannot find a value, it will fill in a none value to the row
    :param df: pd.dataframe | input dataframe
    :param phone_number: str | name of the phone column
    :param address_columns: [str] | list of strings that name the geocode columns
    :param geocode_address: bol | indicates whether addresses should be geocoded
    :param api_key: str | api key for the google API
    :return: np.array, pd.series, pd.series | Returns the preprocesed numpy arry,
    if geocoding is applied it will in addition provide the numeric latitude and
    longitude columns
    """

    # convert all columns to strings
    for a in df.columns:
        df[a] = df[a].astype(str)

    text_processor = Process_text()
    df = df.astype(str).applymap(text_processor.standard_text_normalization)

    if phone_number:
        df[phone_number] = df[phone_number].apply(preprocess_phone)

    #TODO: add e-mail preprocessing.

    if address_columns and geocode_address:
        df = df.fillna('NaN')
        df[['GoogleAddress', 'latitude', 'longitude']] = df[
            address_columns].apply(
            lambda x: pd.Series(geocode_address_api(x, api_key)), axis=1)
        return df.iloc[:,:-2].astype(str).as_matrix(), df.latitude, df.longitude
    else:
        return df.astype(str).as_matrix(), None, None




