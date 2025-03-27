import numpy as np
from scipy import stats
import statsmodels.api as sm
import re
from PyPDF2 import PdfMerger
import pandas as pd
import os
from utils.beh_functions import session_dirs, parseSessionID
from scipy.stats import norm
from datetime import datetime
import ast
import pickle

class opto_metrics:
    def __init__(self, session, data_type):
        """Initialize the object with a DataFrame."""
        session_dir = session_dirs(session)
        unit_tbl_dir = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging_metrics.pkl')
        with open(unit_tbl_dir, 'rb') as f:
            unit_data = pickle.load(f)
        qm_data = unit_data['opto_tagging_df_metrics']

        self.qm_data = qm_data

    def load_unit(self, unit_id):
        return self.qm_data[self.qm_data['unit_id'] == unit_id]

