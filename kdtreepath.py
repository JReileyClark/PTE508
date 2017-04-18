import numpy as np
import pandas as pd
from scipy.spatial import kdtree as kd


surveys = pd.read_excel('/Volumes/SanDisk/Dsurveys for KDTree.xlsx',header=1)
print(surveys.type)