from functools import reduce
import operator
import numpy as np
import json
from os.path import abspath, dirname, join
import datetime

rootdir = dirname(abspath(__file__))

thresholds = {
    43682: {
        'alm_threshold': 1.8,
        'start_date': datetime.datetime(2018,10,29).strftime("%Y-%m-%d"),
    },
    41335: {
        'alm_threshold': 1.8,
        'start_date': datetime.datetime(2016,2,24).strftime("%Y-%m-%d"),
    },
    41240: {
        'alm_threshold': 2.0,
        'start_date': datetime.datetime(2016,1,27).strftime("%Y-%m-%d"),
    },
    39086: {
        'alm_threshold': 1.6,
        'start_date': datetime.datetime(2013,2,25).strftime("%Y-%m-%d"),
    },
    36508: {
        'alm_threshold': 1.3,
        'start_date': datetime.datetime(2010,4,27).strftime("%Y-%m-%d"),
    },
    33105: {
        'alm_threshold': 2.2,
        'start_date': datetime.datetime(2008,6,20).strftime("%Y-%m-%d"),
    },
    27421: {
        'alm_threshold': 2.2,
        'start_date': datetime.datetime(2002,5,4).strftime("%Y-%m-%d"),
    },
    27386: {
        'alm_threshold': 0.8,
        'start_date': datetime.datetime(2002,4,12).strftime("%Y-%m-%d"),
    },
    26997: {
        'alm_threshold': 2.5,
        'start_date': datetime.datetime(2001,12,7).strftime("%Y-%m-%d"),
    },
    25260: {
        'alm_threshold': 1.1,
        'start_date': datetime.datetime(2000,1,4).strftime("%Y-%m-%d"),
    },
    20436: {
        'alm_threshold': 2.1,
        'start_date': datetime.datetime(2000,1,19).strftime("%Y-%m-%d"),
    }
}

# Save thresholds as a JSON file
save_file = join(rootdir, 'conf', 'alm_thresholds.json')
with open(save_file, 'w') as json_file:
    json.dump(thresholds, json_file, indent=2)

print(f"Thresholds saved to {save_file}")