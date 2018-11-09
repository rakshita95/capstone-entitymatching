"""



"""

import re


def preprocess_address(address):
    """

    :param address: a n x m matrix of address columns, including ZIP code,
                    apartment number, etc.
    :return: one address column in a standard format

    """
    # TODO: use an API to get address into a standard format https://pypi.org/project/geopy/
    return None

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


def preprocess_special_fields(df, phone=None, mail=None, address=None):
    """

    :param df:
    :param phone:
    :return:
    """
    if phone:
        df[phone] = df[phone].apply(preprocess_phone)

    return df