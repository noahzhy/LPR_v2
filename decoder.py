import os
import re
import random
import numpy as np


# filter out the error license plate
def is_valid_license_plate(license_plate):
    # remove space
    license_plate = license_plate.replace(' ', '')
    regex = re.compile(r'^([가-힣]{2}\d{2}|\d{2,3})[가-힣]{1}\d{4}$')
    return regex.match(license_plate) is not None


if __name__ == "__main__":
    res = is_valid_license_plate("서울12가1234")
    print(res)
    res = is_valid_license_plate("서울125가1234")
    print(res)
    res = is_valid_license_plate("123가1234")
    print(res)
    res = is_valid_license_plate("12가1234")
    print(res)
