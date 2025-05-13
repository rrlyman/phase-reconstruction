"""
Created on Jul 7, 2018

based upon

"A Non-iterative Method for (Re)Construction of Phase from STFT Magnitude"
Zdenek Prusa, Peter Balazs, Peter L. Sondergaard

@author: richard

to run:
    python PGHI_test.py
or
    python3 PGHI_test.py

"""

import html_table

param_names = ["file_name", "time_scale", "freq_scale"]
table = html_table.HTMLTable(param_names)
