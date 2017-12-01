###parse_meta.py

##a function to parse plexon metadata to save to the RatUnits file

import plxread

"""
This function takes in the path of a .plx file, and returns the relevant
metadata in dictionary format. 
Assumes certain things about file path; namely that it is in the format "BMI_D0x.plx" 
"""
def get_meta(filepath):
	results = {} ##data to return
	##open the file 
	data = plxread.import_file(filepath,AD_channels=range(1,200),import_unsorted=False,
		save_wf=False)
	units = [key for key in data.keys() if key.startswith('sig')]
	lfp_chans = [key for key in data.keys() if key.startswith('AD') and not key.endswith('_ts')]
	events =  [key for key in data.keys() if key.startswith('Event')]
	print("Units: ")
	print(units)
	print("LFP: ")
	print(lfp_chans)
	print("Events ")
	print(events)
