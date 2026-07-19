import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import limits

# Get raw module angles (relative to defined ground plane). Using normal vectors
def get_module_angle(df: pd.dataFrame):

    return x_angle, y_angle, z_angle

# Get distance in xyz between top markers (check if the trial, defined by wire_diam, pvc_spacing, or phi_fb, have the top markers switched)