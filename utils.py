import sys
import time
import sunpy.time
import sunpy.io
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()
engine = create_engine('sqlite:///pore.db', echo=True)
Session = sessionmaker(bind=engine)
session = Session()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        sys.stdout.write(
            '{} : {} ms\n'.format(
                method.__name__, (te - ts) * 1000
            )
        )
        return result
    return timed


@timeit
def parse_time_from_sunpy(header):
    return sunpy.time.parse_time(header['T_OBS'])


@timeit
def get_julian_day_from_astropy_time(astropy_time):
    return astropy_time.jd


@timeit
def get_hmi_jul_day(hmi_image):
    header = sunpy.io.read_file_header(hmi_image, filetype='fits')[1]

    time = parse_time_from_sunpy(header)

    return get_julian_day_from_astropy_time(time)


def _get_julian_time(date_time):
    return sunpy.time.parse_time(date_time.isoformat()).jd


get_julian_time = np.vectorize(_get_julian_time)
