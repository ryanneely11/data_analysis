# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 08:00:12 2014

Just a storage file to store all of the rat units.
format of "animals" is this. Dict keys are animal names, and they refer to a list of the following items:
1) directory path where actual data is stored
2) dictionary of dictionaries, where each entry is a session:sorted units pairing

@author: Ryan
"""


animals = {

"V01":{   

  "BMI_D10.plx":
      {'units':{
      'e1_units':[u'sig019a', u'sig027a'],
      'e2_units':[u'sig032a', u'sig020b'], 
      'Str_units':[u'sig031a', u'sig027b', u'sig024b', u'sig017a', u'sig020a', u'sig022a', u'sig023a', u'sig024a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D09.plx":
      {'units':{
      'e1_units':[u'sig019b', u'sig032a'],
      'e2_units':[u'sig027a', u'sig031a'], 
      'Str_units':[u'sig017a', u'sig020b', u'sig022a', u'sig023a', u'sig024a', u'sig024b', u'sig029a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D08.plx":
      {'units':{
      'e1_units':[u'sig017a', u'sig020a'],
      'e2_units':[u'sig019b', u'sig027a'], 
      'Str_units':[u'sig018a', u'sig032a', u'sig031a', u'sig029a', u'sig024b', u'sig019a', u'sig019c', u'sig020b', u'sig022a', u'sig023a', u'sig024a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D07.plx":
      {'units':{
      'e1_units':[u'sig020a', u'sig027a'],
      'e2_units':[u'sig017a', u'sig018a'], 
      'Str_units':[u'sig019a', u'sig019b', u'sig019c', u'sig020b', u'sig023a', u'sig032a', u'sig031a', u'sig026a', u'sig029a', u'sig024a', u'sig024b', u'sig022a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

    "BMI_D06.plx":
      {'units':{
      'e1_units':[u'sig019b', u'sig020a'],
      'e2_units':[u'sig017a', u'sig027a'], 
      'Str_units':[u'sig019c', u'sig019c', u'sig020b', u'sig022a', u'sig023a', u'sig024a', u'sig024b', u'sig026a', u'sig029a', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},
  
  "BMI_D05.plx":
      {'units':{
      'e1_units':[u'sig004b', u'sig008a'],
      'e2_units':[u'sig016a', u'sig011a'], 
      'Str_units':[u'sig002a', u'sig003a', u'sig003b', u'sig005a', u'sig010a', u'sig013a', u'sig015a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D04.plx":
      {'units':{
      'e1_units':[u'sig020a', u'sig026a'],
      'e2_units':[u'sig021a', u'sig027a'], 
      'Str_units':[u'sig017a', u'sig018a', u'sig019a', u'sig019b', u'sig020b', u'sig022a', u'sig024a', u'sig032a', u'sig031a', u'sig029a', u'sig024b']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D03.plx":
      {'units':{
      'e1_units':[u'sig021a', u'sig032a'],
      'e2_units':[u'sig022a', u'sig026a'], 
      'Str_units':['sig016a', u'sig017a', u'sig018a', u'sig019a', u'sig019b', u'sig020a', u'sig020b', u'sig024a', u'sig027a', u'sig024b', u'sig029a', u'sig031a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D02.plx":
      {'units':{
      'e1_units':[u'sig017a', u'sig032a'],
      'e2_units':[u'sig022a', u'sig021a'], 
      'Str_units':[u'sig016a', u'sig018a', u'sig019a', u'sig019b', u'sig020a', u'sig020b', u'sig024a', u'sig024b', u'sig026a', u'sig027a', u'sig031a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D01.plx":
      {'units':{
      'e1_units':[u'sig001a', u'sig012a'],
      'e2_units':[u'sig009a', u'sig014a'], 
      'Str_units':[u'sig031a', u'sig032a', u'sig024a', u'sig024b', u'sig017a', u'sig020a', u'sig022a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},
},

"V02":{  

  "BMI_D01.plx":
      {'units':{
      'e1_units':[u'sig003a', u'sig012a'],
      'e2_units':[u'sig010a', u'sig004a'], 
      'V1_units':[u'sig002a', u'sig005a', u'sig009a'],
      'Str_units':[u'sig018a', u'sig019a', u'sig020a', u'sig024a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event012'], 't2':['Event011'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D02.plx":
      {'units':{
      'e1_units':[u'sig004a', u'sig003a'],
      'e2_units':[u'sig012a', u'sig002a'], 
      'V1_units':[u'sig001a', u'sig005a', u'sig009a', u'sig010a'],
      'Str_units':[u'sig017a', u'sig018a', u'sig018b', u'sig019a', u'sig020a', u'sig020b', u'sig024a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D03.plx":
      {'units':{
      'e1_units':[u'sig003a', u'sig012a'],
      'e2_units':[u'sig004a', u'sig005a'], 
      'V1_units':[u'sig001a', u'sig002a', u'sig009a', u'sig010a'],
      'Str_units':[u'sig017a', u'sig018a', u'sig018b', u'sig019a', u'sig020a', u'sig020b', u'sig028a', u'sig024a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D04.plx":
      {'units':{
      'e1_units':[u'sig003a', u'sig009a'],
      'e2_units':[u'sig004a', u'sig012a'], 
      'V1_units':[u'sig002a', u'sig005a', u'sig010a'],
      'Str_units':[u'sig019a', u'sig020a', u'sig020b', u'sig024a', u'sig028a', u'sig018a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D05.plx":
      {'units':{
      'e1_units':['sig004a', 'sig003a'],
      'e2_units':['sig002a','sig009a'], 
      'V1_units':['sig005a','sig010a','sig012a'],
      'Str_units':[u'sig018a',u'sig019a', u'sig020a', u'sig020b',u'sig023a', u'sig024a', u'sig028a', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D06.plx":
      {'units':{
      'e1_units':[u'sig004a', u'sig003a'],
      'e2_units':[u'sig002a', u'sig009a'], 
      'V1_units':[u'sig001a', u'sig005a', u'sig010a', u'sig010b', u'sig010c', u'sig012a'],
      'Str_units':[u'sig020a', u'sig020b', u'sig023a', u'sig024a', u'sig028a', u'sig028b', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D07.plx":
      {'units':{
      'e1_units':[u'sig009a', u'sig010a'],
      'e2_units':[u'sig003a', u'sig005a'], 
      'V1_units':[u'sig001a', u'sig004a', u'sig002a', u'sig012a'],
      'Str_units':[u'sig019a', u'sig020b', u'sig020a', u'sig023a', u'sig032a', u'sig028a', u'sig024a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D08.plx":
      {'units':{
      'e1_units':[u'sig010a', u'sig009a'],
      'e2_units':[u'sig003a', u'sig002a'], 
      'V1_units':[u'sig001a', u'sig004a', u'sig005a'],
      'Str_units':[u'sig018a', u'sig019a', u'sig020a',u'sig023a', u'sig024a', u'sig028a', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D09.plx":
      {'units':{
      'e1_units':[u'sig004a', u'sig010a'],
      'e2_units':[u'sig009a', u'sig003a'], 
      'V1_units':[u'sig001a', u'sig002a', u'sig005a', u'sig012a'],
      'Str_units':[u'sig018a', u'sig032a', u'sig028a', u'sig024a', u'sig023a', u'sig020b', u'sig020a', u'sig019a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D10.plx":
      {'units':{
      'e1_units':[u'sig004a', u'sig002a'],
      'e2_units':[u'sig003a', u'sig010a'], 
      'V1_units':[u'sig001a', u'sig005a', u'sig012a', u'sig009a'],
      'Str_units':[u'sig018a', u'sig020a', u'sig019a', u'sig020b', u'sig023a', u'sig024a', u'sig028a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},
},

"V03":{  

  "BMI_D02.plx":
      {'units':{
      'e1_units':[u'sig003a', u'sig013a'],
      'e2_units':[u'sig012a', u'sig002a'], 
      'V1_units':[u'sig004a', u'sig005a', u'sig006a', u'sig008a', u'sig009a', u'sig010a', u'sig010b', u'sig014a', u'sig014b', u'sig015a', u'sig016a'],
      'Str_units':[u'sig017a', u'sig018a', u'sig019a', u'sig021a', u'sig021b', u'sig022a', u'sig023a', u'sig024a', u'sig025a', u'sig026a', u'sig027a', u'sig029a', u'sig030a', u'sig031a', u'sig031b', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D03.plx":
      {'units':{
      'e1_units':[u'sig005a', u'sig014a'],
      'e2_units':[u'sig012a', u'sig006a'], 
      'V1_units':[u'sig002a', u'sig003a', u'sig004a', u'sig008a', u'sig009a', u'sig010a', u'sig010b', u'sig013a', u'sig014b', u'sig015a', u'sig016a'],
      'Str_units':[u'sig017a', u'sig018a', u'sig019a', u'sig021a', u'sig021b', u'sig022a', u'sig023a', u'sig024a', u'sig025a', u'sig025b', u'sig026a', u'sig027a', u'sig029a', u'sig030a', u'sig030b', u'sig031a', u'sig031b', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D04.plx":
      {'units':{
      'e1_units':[u'sig005a', u'sig009a'],
      'e2_units':[u'sig014a', u'sig003a'], 
      'V1_units':[u'sig004a', u'sig006a', u'sig012a', u'sig013a'],
      'Str_units':[u'sig017a', u'sig018a', u'sig018b', u'sig019a', u'sig021a', u'sig021b', u'sig023a', u'sig024a', u'sig025a', u'sig025b', u'sig026a', u'sig029a', u'sig030a', u'sig031a', u'sig031b', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D05.plx":
      {'units':{
      'e1_units':[u'sig005a', u'sig009a'],
      'e2_units':[u'sig003a', u'sig014a'], 
      'V1_units':[u'sig004a', u'sig006a', u'sig010a', u'sig012a', u'sig013a'],
      'Str_units':[u'sig017a', u'sig018a', u'sig018b', u'sig019a', u'sig021a', u'sig023a', u'sig023b', u'sig024a', u'sig025a', u'sig025b', u'sig026a', u'sig029a', u'sig030a', u'sig030b', u'sig031a', u'sig031b', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D06.plx":
      {'units':{
      'e1_units':[u'sig012a', u'sig013a'],
      'e2_units':[u'sig005a', u'sig003a'], 
      'V1_units':[u'sig006a',u'sig007a',u'sig009a',u'sig010a',u'sig014a'],
      'Str_units':[u'sig017a', u'sig018a', u'sig019a', u'sig020a', u'sig023a', u'sig025a', u'sig025b', u'sig026a', u'sig028a', u'sig029b', u'sig030a',u'sig031a',u'sig031b',u'sig032a',u'sig032b']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D07.plx":
      {'units':{
      'e1_units':[u'sig013a', u'sig003a'],
      'e2_units':[u'sig005a', u'sig012a'], 
      'V1_units':[u'sig006a', u'sig009a', u'sig010a', u'sig010b', u'sig014a'],
      'Str_units':[u'sig017a', u'sig018a', u'sig025a', u'sig023a', u'sig030a', u'sig031a', u'sig031b', u'sig032a', u'sig032b', u'sig029a', u'sig028a', u'sig020a', u'sig026a', u'sig030b', u'sig019a', u'sig018b']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D08.plx":
      {'units':{
      'e1_units':[u'sig004a', u'sig016a'],
      'e2_units':[u'sig010a', u'sig012a'], 
      'V1_units':[u'sig010b', u'sig003a', u'sig005a', u'sig006a', u'sig007a', u'sig009a', u'sig013a', u'sig014a', u'sig015a'],
      'Str_units':[u'sig017a', u'sig018a', u'sig018b', u'sig019a', u'sig020a', u'sig023a', u'sig025a', u'sig026a', u'sig029a', u'sig030b', u'sig031b', u'sig032a', u'sig032b', u'sig028a', u'sig030a', u'sig031a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D09.plx":
      {'units':{
      'e1_units':[u'sig004a', u'sig012a'],
      'e2_units':[u'sig005a', u'sig014a'], 
      'V1_units':[u'sig001a', u'sig003a', u'sig006a', u'sig009a', u'sig013a', u'sig015a'],
      'Str_units':[u'sig016a', u'sig017a', u'sig018a', u'sig018b', u'sig019a', u'sig020a', u'sig023a', u'sig025a', u'sig026a', u'sig027a', u'sig027b', u'sig028a', u'sig029a', u'sig030a', u'sig030b', u'sig031a', u'sig032a', u'sig032b']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D10.plx":
      {'units':{
      'e1_units':[u'sig004a', u'sig012a'],
      'e2_units':[u'sig005a', u'sig014a'], 
      'V1_units':[u'sig003a', u'sig006a', u'sig009a', u'sig010a', u'sig013a'],
      'Str_units':[u'sig018a', u'sig018b', u'sig019a', u'sig020a', u'sig023a', u'sig024a', u'sig025a', u'sig026a', u'sig028a', u'sig029a', u'sig030b', u'sig030a', u'sig031a', u'sig032a', u'sig032b']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

},

"V04":{  

  "BMI_D01.plx":
      {'units':{
      'e1_units':[u'sig012a', u'sig014a'],
      'e2_units':[u'sig013a', u'sig015a'], 
      'V1_units':[u'sig006a', u'sig007a', u'sig008a', u'sig010a', u'sig016a'],
      'Str_units':[u'sig017a', u'sig017b', u'sig018a', u'sig019a', u'sig020a', u'sig020b', u'sig021a', u'sig021b', u'sig022a', u'sig023a', u'sig023b', u'sig024a', u'sig025a', u'sig025b', u'sig026a', u'sig027a', u'sig028a', u'sig028b', u'sig029a', u'sig029b', u'sig029c', u'sig031a', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event012'], 't2':['Event011'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D02.plx":
      {'units':{
      'e1_units':[u'sig006b', u'sig014a'],
      'e2_units':[u'sig012a', u'sig013a'], 
      'V1_units':[u'sig006a', u'sig007a', u'sig008a', u'sig010a', u'sig011a',u'sig015a',u'sig016a'],
      'Str_units':[u'sig017a', u'sig017b', u'sig018a', u'sig019a', u'sig019b', u'sig020a',u'sig020b', u'sig021a', u'sig022a', u'sig023a', u'sig023b', u'sig024a', u'sig025a', u'sig025b', u'sig026a', u'sig027a', u'sig028a', u'sig028b', u'sig029a', u'sig029b', u'sig029c', u'sig031a', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},


  "BMI_D03.plx":
      {'units':{
      'e1_units':[u'sig006a', u'sig015a'],
      'e2_units':[u'sig012a', u'sig013a'], 
      'V1_units':[u'sig007a', u'sig008a', u'sig010a', u'sig014a', u'sig015b', u'sig016a'],
      'Str_units':[u'sig017a', u'sig017b', u'sig018a', u'sig019a', u'sig019b', u'sig020a', u'sig020b', u'sig021a', u'sig022a', u'sig023a', u'sig023b', u'sig024a', u'sig025a', u'sig025b', u'sig026a', u'sig027a', u'sig028a', u'sig028b', u'sig029a', u'sig029b', u'sig031a', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},


  "BMI_D04.plx":
      {'units':{
      'e1_units':[u'sig007a', u'sig015a'],
      'e2_units':[u'sig013a', u'sig014a'], 
      'V1_units':[u'sig006a', u'sig008a'],
      'Str_units':[u'sig016a', u'sig017a', u'sig017b', u'sig018a', u'sig019a', u'sig025b', u'sig026a', u'sig027a', u'sig028a', u'sig028b', u'sig029a', u'sig025a', u'sig023b', u'sig032a', u'sig031a', u'sig029b', u'sig019b', u'sig020a', u'sig020b', u'sig021a', u'sig022a', u'sig023a', u'sig024a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},


  "BMI_D05.plx":
      {'units':{
      'e1_units':[u'sig007a', u'sig015a'],
      'e2_units':[u'sig014a', u'sig013a'], 
      'V1_units':[u'sig006a', u'sig008a', u'sig011a', u'sig016a'],
      'Str_units':[u'sig017a', u'sig017b', u'sig017c', u'sig018a', u'sig018b', u'sig019a', u'sig019b', u'sig020a', u'sig020b', u'sig021a', u'sig023a', u'sig022a', u'sig023b', u'sig024a', u'sig025a', u'sig025b', u'sig027a', u'sig028a', u'sig029a', u'sig031a', u'sig032a', u'sig029b', u'sig028b', u'sig027b', u'sig026a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},


  "BMI_D06.plx":
      {'units':{
      'e1_units':[u'sig007a', u'sig015a'],
      'e2_units':[u'sig014a', u'sig013a'], 
      'V1_units':[u'sig006a', u'sig008a', u'sig016a'],
      'Str_units':[u'sig017a', u'sig017b', u'sig018a', u'sig018b', u'sig019a', u'sig019b', u'sig020a', u'sig020b', u'sig021a', u'sig023a', u'sig022a', u'sig023b', u'sig024a', u'sig025a', u'sig025b', u'sig026a', u'sig027a',u'sig027b', u'sig028a', u'sig028b', u'sig029a', u'sig029b', u'sig031a', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D07.plx":
      {'units':{
      'e1_units':[u'sig007a', u'sig015a'],
      'e2_units':[u'sig013a', u'sig014a'], 
      'V1_units':[u'sig006a', u'sig008a', u'sig016a'],
      'Str_units':[u'sig018b', u'sig020b', u'sig020a', u'sig019b', u'sig021a', u'sig022a', u'sig023a', u'sig025a', u'sig024a', u'sig023b', u'sig027a', u'sig027b', u'sig028a', u'sig026a', u'sig025b', u'sig019a', u'sig017b', u'sig017a', u'sig018a', u'sig028b', u'sig029a', u'sig029b', u'sig031a', u'sig032a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D08.plx":
      {'units':{
      'e1_units':[u'sig008a', u'sig015a'],
      'e2_units':[u'sig016a', u'sig014a'], 
      'V1_units':[u'sig006a', u'sig007a', u'sig013a'],
      'Str_units':[u'sig017a', u'sig017b', u'sig018a', u'sig018b', u'sig019a', u'sig019b', u'sig020a', u'sig020b', u'sig021a', u'sig022a', u'sig023b', u'sig025a', u'sig025b', u'sig027b', u'sig029b', u'sig031a', u'sig032a', u'sig029a', u'sig028a', u'sig026a', u'sig024a', u'sig023a', u'sig027a', u'sig028b']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D09.plx":
      {'units':{
      'e1_units':[u'sig008a', u'sig015a'],
      'e2_units':[u'sig016a', u'sig014a'], 
      'V1_units':[u'sig006a', u'sig007a', u'sig013a'],
      'Str_units':[u'sig017a', u'sig017b', u'sig018a', u'sig018b', u'sig019a', u'sig019b', u'sig020a', u'sig020b', u'sig021a', u'sig022a', u'sig023b', u'sig025a', u'sig025b', u'sig027b', u'sig029b', u'sig031a', u'sig032a', u'sig029a', u'sig028a',  u'sig028b', u'sig026a', u'sig024a', u'sig023a', u'sig027a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D10.plx":
      {'units':{
      'e1_units':[u'sig006a', u'sig015a'],
      'e2_units':[u'sig014a', u'sig016a'], 
      'V1_units':[u'sig007a', u'sig008a', u'sig013a'],
      'Str_units':[u'sig017a', u'sig017b', u'sig018a', u'sig018b', u'sig019a', u'sig019b', u'sig020a', u'sig020b', u'sig021a', u'sig022a', u'sig023a', u'sig023b', u'sig025b', u'sig028a', u'sig028b', u'sig029a', u'sig029b', u'sig031a', u'sig032a', u'sig024a', u'sig025a', u'sig026a', u'sig027a', u'sig027b']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

},

"V05":{  

  "BMI_D01.plx":
      {'units':{
      'e1_units':[u'sig007a', u'sig008a'],
      'e2_units':[u'sig016a'], 
      'V1_units':[],
      'Str_units':[u'sig017a', u'sig018a', u'sig020a', u'sig021a', u'sig022a', u'sig024a', u'sig024b', u'sig027a', u'sig029a', u'sig030a', u'sig031a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event012'], 't2':['Event011'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D02.plx":
      {'units':{
      'e1_units':[u'sig008a', u'sig016a'],
      'e2_units':[u'sig007a', u'sig007b'], 
      'V1_units':[],
      'Str_units':[u'sig017a', u'sig018a', u'sig018b', u'sig020a', u'sig021a', u'sig022a', u'sig023a', u'sig024a', u'sig024b', u'sig027a', u'sig029a', u'sig030a', u'sig031a', u'sig017b',u'sig023b',u'sig031b']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event012'], 't2':['Event011'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D03.plx":
      {'units':{
      'e1_units':[u'sig008a', u'sig007a'],
      'e2_units':[u'sig016a', u'sig012a'], 
      'V1_units':[],
      'Str_units':[u'sig017a', u'sig018a', u'sig020a', u'sig021a', u'sig022a', u'sig024a', u'sig027a', u'sig029a', u'sig031a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D04.plx":
      {'units':{
      'e1_units':[u'sig002a', u'sig008a'],
      'e2_units':[u'sig012a', u'sig016a'], 
      'V1_units':[],
      'Str_units':[u'sig017a', u'sig017b', u'sig020a', u'sig022a', u'sig023a', u'sig023b', u'sig024a', u'sig024b', u'sig027a', u'sig029a', u'sig030a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D05.plx":
      {'units':{
      'e1_units':['sig002a', u'sig008a'],
      'e2_units':[u'sig012a', u'sig016a'], 
      'V1_units':[],
      'Str_units':[u'sig017a', u'sig017b', u'sig018a', u'sig023a', u'sig023b', u'sig024a', u'sig024b', u'sig026a', u'sig030a', u'sig031a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D06.plx":
      {'units':{
      'e1_units':[u'sig002a', u'sig008a'],
      'e2_units':[u'sig012a', u'sig016a'], 
      'V1_units':[],
      'Str_units':[u'sig017a', u'sig017b', u'sig018a', u'sig023a', u'sig027a', u'sig029a', u'sig024b', u'sig020a', u'sig020b', u'sig024a', u'sig026a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D07.plx":
      {'units':{
      'e1_units':[u'sig005a', u'sig008a'],
      'e2_units':[u'sig012a', u'sig016a'], 
      'V1_units':[],
      'Str_units':[u'sig017a', u'sig017b', u'sig018a', u'sig019a', u'sig020a', u'sig023a', u'sig020b', u'sig023b', u'sig024a', u'sig026a', u'sig027a', u'sig027b', u'sig029a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D08.plx":
      {'units':{
      'e1_units':[u'sig005a', u'sig008a'],
      'e2_units':[u'sig012a', u'sig016a'], 
      'V1_units':[],
      'Str_units':[u'sig017a', u'sig020a', u'sig020b', u'sig021a', u'sig023a', u'sig023b', u'sig024a', u'sig026a', u'sig027a', u'sig029a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

  "BMI_D09.plx":
      {'units':{
      'e1_units':[u'sig005a', u'sig008a'],
      'e2_units':[u'sig012a', u'sig016a'], 
      'V1_units':[],
      'Str_units':[u'sig017a', u'sig021a', u'sig024a', u'sig026a', u'sig027a', u'sig029a', u'sig023b', u'sig023a', u'sig020a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},


  "BMI_D10.plx":
      {'units':{
      'e1_units':[u'sig005a', u'sig008a'],
      'e2_units':[u'sig012a', u'sig016a'], 
      'V1_units':[],
      'Str_units':[u'sig017a', u'sig019a', u'sig020a', u'sig021a', u'sig023a', u'sig023b', u'sig024a', u'sig029a', u'sig027a', u'sig026a']},
       'lfp':{
       'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 't2':['Event012'], 'miss':['Event010']},
       'control_cells':'V1'},

},

"R7":{
  
  "BMI_D05.plx":{
  'units':{
  'e1_units':['sig006a','sig011a'],
  'e2_units':['sig025a','sig026a'], 
  'V1_units':['sig001a','sig009a','sig014a','sig014b','sig015a']},
       'lfp':{
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 'miss':['Event015']},
       'control_cells':'V1'},
  
  "BMI_D06.plx":{
  'units':{
  'e1_units':['sig006a','sig014a'],
  'e2_units':['sig025a','sig026a'], 
  'V1_units':['sig011a','sig028a']},
       'lfp':{
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 'miss':['Event015']},
       'control_cells':'V1'},
  
},
     
"R8":{
 
 "BMI_D05.plx":{
 'units':{
 'e1_units':['sig005a','sig016a'],
 'e2_units':['sig023a','sig031a'], 
 'V1_units':['sig014a']},
       'lfp':{
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 'miss':['Event015']},
       'control_cells':'V1'},
 
 "BMI_D06.plx":{
 'units':{
 'e1_units':['sig005a','sig014a'],
 'e2_units':['sig019a','sig031a'], 
 'V1_units':['sig020a','sig022a']},
       'lfp':{
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 'miss':['Event015']},
       'control_cells':'V1'},
 
 "BMI_D07.plx":{
 'units':{
 'e1_units':['sig005a','sig014a'],
 'e2_units':['sig029a','sig031a'], 
 'V1_units':['sig020a','sig021a']},
       'lfp':{
       'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016']},
       'events':{'t1':['Event011'], 'miss':['Event015']},
       'control_cells':'V1'},
},

"R11":{

  "BMI_D01.plx":{
  'units':{
  'e2_units':['sig034a','sig042a'],
  'e1_units':['sig036a','sig036b'], 
  'PLC_units':['sig001a','sig005a','sig008a','sig009a','sig012a','sig013a','sig014a','sig015a'], 
  'V1_units':['sig019b','sig019a','sig020a','sig022a','sig023a','sig025a','sig028a','sig029a','sig030b','sig032a'], 
  'Str_units':['sig037a','sig043a','sig035a','sig046a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'Str'},

  "BMI_D02.plx":{
  'units':{
  'e2_units':['sig034a','sig042a'],
  'e1_units':['sig036a','sig036b'], 
  'PLC_units':['sig001a','sig002a','sig008a','sig009a','sig010a','sig012a'],
  'V1_units':['sig018a','sig019b','sig021a','sig025a','sig026a','sig029a'], 
  'Str_units':['sig035a','sig043a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'Str'},

  "BMI_D03.plx":{
  'units':{
  'e2_units':['sig034a','sig042a'],
  'e1_units':['sig036a','sig036b'], 
  'PLC_units':['sig001a','sig002a','sig008a','sig009a','sig010a','sig012a','sig016a'],
  'V1_units':['sig025a','sig026a','sig027a','sig029a','sig030a','sig030b','sig032a'], 
  'Str_units':['sig048a','sig046a','sig045a','sig044a','sig043a','sig037a','sig035a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'Str'},

  "BMI_D04.plx":{
  'units':{
  'e2_units':['sig025a','sig030b'],
  'e1_units':['sig018a','sig020a'], 
  'PLC_units':['sig001a','sig002a','sig008a','sig009a','sig010a','sig012a','sig013a','sig016a'], 
  'V1_units':['sig019a','sig022a','sig026a','sig027a','sig029a','sig030a'], 
  'Str_units':['sig034a','sig035a','sig035b','sig036a','sig036b','sig042a','sig043a','sig043b','sig044a','sig045a','sig046a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D05.plx":{
  'units':{
  'e2_units':['sig019a','sig031a'],
  'e1_units':['sig018a','sig020a'], 
  'PLC_units':['sig001a','sig002a','sig008a','sig009a','sig010a','sig012a','sig013a','sig016a'],
  'V1_units':['sig025a','sig026a','sig027a','sig029a','sig031a','sig032a'], 
  'Str_units':['sig048a','sig046a','sig045a','sig044a','sig043a','sig043b','sig037a','sig034a','sig036a','sig036b']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D06.plx":{
  'units':{
  'e2_units':['sig019a','sig031a'],
  'e1_units':['sig025a','sig026a'], 
  'PLC_units':['sig001a','sig008a','sig009a','sig010a','sig012a','sig014a','sig016a'],
  'V1_units':['sig020a','sig027a','sig029a'], 
  'Str_units':['sig048a','sig046a','sig045a','sig044a','sig043a','sig043b','sig036a','sig036b','sig034a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D07.plx":{
  'units':{
  'e2_units':['sig025a','sig029a'],
  'e1_units':['sig018a','sig020a'], 
  'PLC_units':['sig001a','sig002a','sig008a','sig009a','sig010a','sig012a','sig014a'],
  'V1_units':['sig026a','sig027a','sig020a','sig031a'], 
  'Str_units':['sig048a','sig046a','sig045a','sig044a','sig043a','sig037a','sig035a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D08.plx":{
  'units':{
  'e2_units':['sig025a','sig029a'],
  'e1_units':['sig018a','sig019a'], 
  'PLC_units':['sig001b','sig002a','sig008a','sig009a','sig010a','sig012a','sig013a','sig014a','sig014b','sig016a'],
  'V1_units':['sig026a','sig020a','sig031a'], 
  'Str_units':['sig048a','sig046a','sig045a','sig044a','sig043a','sig043b','sig037a','sig034a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D09.plx":{
  'units':{
  'e2_units':['sig020a','sig025a'],
  'e1_units':['sig018a','sig019a'], 
  'PLC_units':['sig001a','sig001b','sig002a','sig007a','sig008a','sig009a','sig010a','sig010b','sig012a','sig013a','sig014a','sig016a'],
  'V1_units':['sig029a','sig031a'], 
  'Str_units':['sig048a','sig046a','sig045a','sig044a','sig043a','sig043b','sig042a','sig036a','sig036b','sig037a','sig035a','sig034a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D10.plx":{
  'units':{
  'e2_units':['sig017a','sig027a'],
  'e1_units':['sig026a','sig028a'], 
  'PLC_units':['sig001a','sig002a','sig007a','sig008a','sig009a','sig010a','sig012a','sig015a','sig014a'],
  'V1_units':['sig018a','sig020a','sig025a','sig031a'], 
  'Str_units':['sig044a','sig043a','sig041a','sig035a','sig034a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

},

"R13":{

  "BMI_D01.plx":{
  'units':{
  'e2_units':['sig020a','sig021a'],
  'e1_units':['sig019a','sig027a'], 
  'V1_units':['sig022a','sig019a','sig024a','sig025a','sig026a','sig029a','sig030a','sig031a','sig032a'], 
  'PLC_units':['sig001a','sig002a','sig002b','sig004a','sig007a','sig008a','sig013a','sig012a','sig010a','sig015a'], 
  'Str_units':['sig033a','sig034a','sig036a','sig040a','sig043a','sig044a','sig047a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D02.plx":{
  'units':{
  'e2_units':['sig020a','sig032a'],
  'e1_units':['sig022a','sig027a'], 
  'V1_units':['sig017a','sig019a','sig021a','sig025a','sig026a','sig029a','sig030a','sig031a'], 
  'PLC_units':['sig001a','sig002a','sig002b','sig003a','sig004a','sig006a','sig007a','sig008a','sig009a','sig012a','sig010a'], 
  'Str_units':['sig035a','sig038a','sig039a','sig040a','sig043a','sig046a','sig047a','sig048a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D03.plx":{
  'units':{
  'e2_units':['sig021a','sig032a'],
  'e1_units':['sig022a','sig026a'], 
  'V1_units':['sig020a','sig023a','sig025a','sig027a','sig029a','sig031a'], 
  'PLC_units':['sig001a','sig002a','sig003a','sig006a','sig007a','sig007b','sig008a','sig009a','sig014a','sig012a','sig010a','sig016a'], 
  'Str_units':['sig035a','sig036a','sig043a','sig046a','sig047a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D04.plx":{
  'units':{
  'e2_units':['sig021a','sig032a'],
  'e1_units':['sig022a','sig026a'], 
  'V1_units':['sig020a','sig024a','sig029a','sig027a','sig031a'], 
  'PLC_units':['sig001a','sig002a','sig003a','sig004a','sig005a','sig006a','sig007a','sig007b','sig008a','sig013a','sig012a','sig010a','sig016a'], 
  'Str_units':['sig035a','sig036a','sig042a','sig043a','sig047a','sig048a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D05.plx":{
  'units':{
  'e2_units':['sig024a','sig029a'],
  'e1_units':['sig022a','sig031a'], 
  'V1_units':['sig020a','sig021a','sig027a','sig032a'], 
  'PLC_units':['sig001a','sig002a','sig003a','sig004a','sig004b','sig005a','sig006a','sig007a','sig007b','sig008a','sig013a','sig012a','sig010a','sig016a'], 
  'Str_units':['sig035a','sig039b','sig040a','sig042a','sig043a','sig043b','sig045a','sig047a','sig048a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D06.plx":{
  'units':{
  'e2_units':['sig024a','sig029a'],
  'e1_units':['sig021a','sig031a'], 
  'V1_units':['sig022a','sig026a','sig027a'], 
  'PLC_units':['sig001a','sig001b','sig003a','sig004a','sig005a','sig006a','sig007a','sig008a','sig009a','sig012a','sig010a','sig016a'], 
  'Str_units':['sig035a','sig036a','sig040a','sig042a','sig043a','sig043b','sig045a','sig047a','sig048a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D07.plx":{
  'units':{
  'e2_units':['sig024a','sig032a'],
  'e1_units':['sig021a','sig031a'], 
  'V1_units':['sig020a','sig029a','sig030a','sig026a'], 
  'PLC_units':['sig004a','sig005a','sig006a','sig007a','sig008a','sig010a','sig016a'], 
  'Str_units':['sig035a','sig036a','sig037a','sig039a','sig040a','sig042a','sig043a','sig043b','sig045a','sig047a','sig048a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D08.plx":{
  'units':{
  'e2_units':['sig021a','sig024a'],
  'e1_units':['sig020a','sig029a'], 
  'V1_units':['sig018a','sig022a','sig026a','sig030a','sig031a','sig032a'], 
  'PLC_units':['sig001a','sig006a','sig007a','sig008a','sig010a','sig012a'], 
  'Str_units':['sig035a','sig040a','sig042a','sig043b','sig044a','sig045a','sig046a','sig047a','sig048a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D09.plx":{
  'units':{
  'e2_units':['sig021a','sig032a'],
  'e1_units':['sig020a','sig029a'], 
  'V1_units':['sig026a','sig030a','sig031a'], 
  'PLC_units':['sig001a','sig006a','sig007a','sig008a','sig010a','sig012a'], 
  'Str_units':['sig035a','sig040a','sig042a','sig044a','sig045a','sig046a','sig047a','sig048a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D10.plx":{
  'units':{
  'e2_units':['sig024a','sig032a'],
  'e1_units':['sig020a','sig017a'], 
  'V1_units':['sig021a','sig030a','sig029a'], 
  'PLC_units':['sig004a','sig005a','sig011a','sig012a','sig016a'], 
  'Str_units':['sig035a','sig042a','sig043a','sig039a','sig047a']},
        'lfp':{
        'V1_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'], 
        'PLC_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'], 
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event012'], 'miss':['Event015'], 't2':['Event011']},
        'control_cells':'V1'},
},
    
"V11":{

  "BMI_D01.plx":{
  'units':{
  'e2_units':['sig009a', 'sig013a'],
  'e1_units':['sig005a', 'sig002a'], 
  'V1_units':['sig001a', 'sig015a', 'sig010a', 'sig013a', 'sig016a', 'sig006a', 'sig014a'],  
  'Str_units':['sig017a', 'sig017b', 'sig030a', 'sig028a', 'sig022a', 'sig018a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'],  
        'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D02.plx":{
  'units':{
  'e2_units':['sig002a', 'sig014a'],
  'e1_units':['sig005a', 'sig001a'], 
  'V1_units':['sig009a', 'sig010a', 'sig009b', 'sig010b', 'sig013a', 'sig013b', 'sig015a', 'sig016a', 'sig006a', 'sig006b'],  
  'Str_units':['sig017a', 'sig017b', 'sig030a', 'sig032a', 'sig031a', 'sig025a', 'sig024a', 'sig023a', 'sig022a', 'sig021b', 'sig019a', 'sig020a', 'sig021a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'],  
        'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
        'events':{'t1':['Event012'], 'miss':['Event010'], 't2':['Event011']},
        'control_cells':'V1'},

  "BMI_D03.plx":{
  'units':{
  'e2_units':['sig002a', 'sig014a'],
  'e1_units':['sig005a', 'sig001a'], 
  'V1_units':['sig002b', 'sig009a', 'sig006b', 'sig009b', 'sig006a', 'sig013a', 'sig010a', 'sig013b', 'sig015a', 'sig016a'],  
  'Str_units':['sig017a', 'sig018a', 'sig031a', 'sig030a', 'sig028a', 'sig023a', 'sig025a', 'sig022a', 'sig021a', 'sig020a', 'sig019a', 'sig017b']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'],  
        'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D04.plx":{
  'units':{
  'e2_units':['sig002a', 'sig014a'],
  'e1_units':['sig001a', 'sig005a'], 
  'V1_units':['sig001b', 'sig010a', 'sig013a', 'sig013b', 'sig016a', 'sig015a', 'sig009a', 'sig006a', 'sig009b', 'sig006b'],  
  'Str_units':['sig017a', 'sig020a', 'sig032a', 'sig031a', 'sig024a', 'sig023a', 'sig022a', 'sig021b', 'sig021a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'],  
        'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D05.plx":{
  'units':{
  'e2_units':['sig002a', 'sig014a'],
  'e1_units':['sig001a', 'sig005a'], 
  'V1_units':['sig013b', 'sig015a', 'sig016a', 'sig013a', 'sig012a', 'sig010a', 'sig011a', 'sig009b', 'sig009a', 'sig006b', 'sig006a', 'sig004a', 'sig003a', 'sig001b', 'sig009c'],  
  'Str_units':['sig020a', 'sig021a', 'sig025a', 'sig029a', 'sig031a', 'sig032a'], 'e2_units': ['sig002a', 'sig014a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'],  
        'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D06.plx":{
  'units':{
  'e2_units':['sig002a', 'sig014a'],
  'e1_units':['sig001a', 'sig005a'], 
  'V1_units':['sig006b', 'sig009a', 'sig009b', 'sig010a', 'sig011a', 'sig010b', 'sig013a', 'sig013b', 'sig015a', 'sig016a', 'sig016b', 'sig006a', 'sig005b', 'sig004a', 'sig003b', 'sig003a'],  
  'Str_units':['sig018a', 'sig020a', 'sig017a', 'sig021a', 'sig022a', 'sig032a', 'sig025a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'],  
        'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D07.plx":{
  'units':{
  'e2_units':['sig002a', 'sig014a'],
  'e1_units':['sig001a', 'sig005a'], 
  'V1_units':['sig003a', 'sig004a', 'sig006a', 'sig010a', 'sig011a', 'sig013a', 'sig016a', 'sig015a'],  
  'Str_units':['sig018a', 'sig018b', 'sig017a', 'sig020a', 'sig021a', 'sig029a', 'sig032a', 'sig030a', 'sig026b', 'sig026a', 'sig025a', 'sig023a', 'sig022a', 'sig019a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'],  
        'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D08.plx":{
  'units':{
  'e2_units':['sig002a', 'sig014a'],
  'e1_units':['sig001a', 'sig005a'], 
  'V1_units':['sig003a', 'sig016a', 'sig015a', 'sig013a', 'sig010a', 'sig011a', 'sig009a', 'sig004a', 'sig006a'],  
  'Str_units':['sig029a', 'sig030a', 'sig026a', 'sig026b', 'sig025a', 'sig022a', 'sig021a', 'sig020a', 'sig019a', 'sig017a', 'sig018a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'],  
        'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D09.plx":{
  'units':{
  'e2_units':['sig002a', 'sig014a'],
  'e1_units':['sig001a', 'sig005a'], 
  'V1_units':['sig003a', 'sig006a', 'sig009a', 'sig010a', 'sig013a', 'sig015a', 'sig016a'],  
  'Str_units':['sig017a', 'sig026b', 'sig029a', 'sig031a', 'sig026a', 'sig025a', 'sig024b', 'sig024a', 'sig021a', 'sig022a', 'sig020a', 'sig019a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'],  
        'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},


  "BMI_D10.plx":{
  'units':{
  'e2_units':['sig002a', 'sig014a'],
  'e1_units':['sig001a', 'sig005a'], 
  'V1_units':['sig003a', 'sig016a', 'sig016b', 'sig015a', 'sig006a', 'sig009a', 'sig010a', 'sig013a'],  
  'Str_units':['sig017a', 'sig019a', 'sig031a', 'sig029a', 'sig026b', 'sig025a', 'sig026a', 'sig022a', 'sig024a', 'sig020a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016'],  
        'Str_lfp':['AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},
},
    
"V13":{

  "BMI_D01.plx":{
  'units':{
  'e2_units':['sig018a', 'sig026a'],
  'e1_units':['sig005a', 'sig004a'], 
  'V1_units':['sig012a', 'sig001a', 'sig010a', 'sig010b', 'sig002a', 'sig020a', 'sig013a', 'sig003a', 'sig027a', 'sig016a', 'sig001b', 'sig028a', 'sig011a', 'sig025a', 'sig009a', 'sig017a'],  
  'Str_units':['sig043a', 'sig040a', 'sig034a', 'sig037a', 'sig036a', 'sig042a', 'sig041a', 'sig038a', 'sig047a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016','AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'],  
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D02.plx":{
  'units':{
  'e2_units':['sig018a', 'sig019a'],
  'e1_units':['sig004a', 'sig005a'], 
  'V1_units':['sig001a', 'sig002a', 'sig003a', 'sig010a', 'sig013a', 'sig025a', 'sig027a', 'sig026a', 'sig011a', 'sig016a', 'sig012a', 'sig017a', 'sig020a'],  
  'Str_units':['sig034a', 'sig036a', 'sig044a', 'sig041a', 'sig039a', 'sig038a', 'sig037a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016','AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'],  
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D03.plx":{
  'units':{
  'e2_units':['sig013a', 'sig018a'],
  'e1_units':['sig004a', 'sig005a'], 
  'V1_units':['sig003a', 'sig009a', 'sig010a', 'sig026a', 'sig025a', 'sig019a', 'sig019b', 'sig027a', 'sig011a', 'sig002a', 'sig012a', 'sig001a', 'sig016a', 'sig017a'],  
  'Str_units':['sig034a', 'sig036a', 'sig048a', 'sig038a', 'sig039a', 'sig041a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016','AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'],  
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D04.plx":{
  'units':{
  'e2_units':['sig013a', 'sig018a'],
  'e1_units':['sig004a', 'sig005a'], 
  'V1_units':['sig001a', 'sig002a', 'sig003a', 'sig009a', 'sig010a', 'sig010b', 'sig011a', 'sig026a', 'sig027a', 'sig025a', 'sig017a', 'sig016a', 'sig014a', 'sig012a', 'sig019a'],  
  'Str_units':['sig038a', 'sig035a', 'sig034a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016','AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'],  
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D05.plx":{
  'units':{
  'e2_units':['sig013a', 'sig017a'],
  'e1_units':['sig012a', 'sig004a'], 
  'V1_units':['sig001a', 'sig005a', 'sig0019a', 'sig010a', 'sig002a', 'sig018a', 'sig027a', 'sig016a', 'sig011b', 'sig011a', 'sig025a', 'sig009a', 'sig014a', 'sig026a'],  
  'Str_units':['sig038a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016','AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'],  
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},

  "BMI_D06.plx":{
  'units':{
  'e2_units':['sig013a', 'sig017a'],
  'e1_units':['sig010a', 'sig014a'], 
  'V1_units':['sig002a', 'sig003a', 'sig026a', 'sig025a', 'sig019a', 'sig012a', 'sig013b', 'sig011a', 'sig001a', 'sig001b', 'sig004a', 'sig005a', 'sig009a'],  
  'Str_units':['sig034a', 'sig038a']},
        'lfp':{
        'V1_lfp':['AD001','AD002','AD003','AD004','AD005','AD006','AD007','AD008','AD009','AD010','AD011','AD012','AD013','AD014','AD015','AD016','AD017','AD018','AD019','AD020','AD021','AD022','AD023','AD024','AD025','AD026','AD027','AD028','AD029','AD030','AD031','AD032'],  
        'Str_lfp':['AD033','AD034','AD035','AD036','AD037','AD038','AD039','AD040','AD041','AD042','AD043','AD044','AD045','AD046','AD047','AD048']},
        'events':{'t1':['Event011'], 'miss':['Event010'], 't2':['Event012']},
        'control_cells':'V1'},
},
    }

