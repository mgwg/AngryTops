training_dir = 'CheckPoints/training_scrap'

column_names = ["runNumber", "eventNumber", "weight", "jets_n", "bjets_n",
"lep Px", "lep Py", "lep Pz", "lep E", "met_met", "met_phi",
"jet0 P_x",  "jet0 P_y",  "jet0 P_z",  "jet0 E",  "jet0 M",  "jet0 BTag",
"jet1 P_x",  "jet1 P_y",  "jet1 P_z",  "jet1 E",  "jet1 M",  "jet1 BTag",
"jet2 P_x",  "jet2 P_y",  "jet2 P_z",  "jet2 E",  "jet2 M",  "jet2 BTag",
"jet3 P_x",  "jet3 P_y",  "jet3 P_z",  "jet3 E",  "jet3 M",  "jet3 BTag",
"jet4 P_x",  "jet4 P_y",  "jet4 P_z",  "jet4 E",  "jet4 M",  "jet4 BTag",
"target_W_had_Px", "target_W_had_Py", "target_W_had_Pz", "target_W_had_E", "target_W_had_M",
"target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz", "target_W_lep_E", "target_W_lep_M",
"target_b_had_Px", "target_b_had_Py", "target_b_had_Pz", "target_b_had_E", "target_b_had_M",
"target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz", "target_b_lep_E", "target_b_lep_M",
"target_t_had_Px", "target_t_had_Py", "target_t_had_Pz", "target_t_had_E", "target_t_had_M",
"target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz", "target_t_lep_E", "target_t_lep_M",
"lep Pt", "lep Eta", "lep Phi",
"jet0 Pt", "jet0 Eta", "jet0 Phi",
"jet1 Pt", "jet1 Eta", "jet1 Phi",
"jet2 Pt", "jet2 Eta", "jet2 Phi",
"jet3 Pt", "jet3 Eta", "jet3 Phi",
"jet4 Pt", "jet4 Eta", "jet4 Phi",
"target_W_had_Pt", "target_W_had_Eta", "target_W_had_Phi",
"target_W_lep_Pt", "target_W_lep_Eta", "target_W_lep_Phi",
"target_b_had_Pt", "target_b_had_Eta", "target_b_had_Phi",
"target_b_lep_Pt", "target_b_lep_Eta", "target_b_lep_Phi",
"target_t_had_Pt", "target_t_had_Eta", "target_t_had_Phi",
"target_t_lep_Pt", "target_t_lep_Eta", "target_t_lep_Phi"
]

input_columns = [
"lep Px", "lep Py", "lep Pz", "lep E", "met_met", "met_phi",
"jet0 P_x",  "jet0 P_y",  "jet0 P_z",  "jet0 E",  "jet0 M",  "jet0 BTag",
"jet1 P_x",  "jet1 P_y",  "jet1 P_z",  "jet1 E",  "jet1 M",  "jet1 BTag",
"jet2 P_x",  "jet2 P_y",  "jet2 P_z",  "jet2 E",  "jet2 M",  "jet2 BTag",
"jet3 P_x",  "jet3 P_y",  "jet3 P_z",  "jet3 E",  "jet3 M",  "jet3 BTag",
"jet4 P_x",  "jet4 P_y",  "jet4 P_z",  "jet4 E",  "jet4 M",  "jet4 BTag",
]

output_columns = [
"target_W_had_Px", "target_W_had_Py", "target_W_had_Pz", "target_W_had_E",
"target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz", "target_W_lep_E",
"target_b_had_Px", "target_b_had_Py", "target_b_had_Pz", "target_b_had_E",
"target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz", "target_b_lep_E",
"target_t_had_Px", "target_t_had_Py", "target_t_had_Pz", "target_t_had_E",
"target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz", "target_t_lep_E",
]

output_columns_ptetaphiE = [
"target_W_had_Pt", "target_W_had_Eta", "target_W_had_Phi", "target_W_had_E",
"target_W_lep_Pt", "target_W_lep_Eta", "target_W_lep_Phi", "target_W_lep_E",
"target_b_had_Pt", "target_b_had_Eta", "target_b_had_Phi", "target_b_had_E",
"target_b_lep_Pt", "target_b_lep_Eta", "target_b_lep_Phi", "target_b_lep_E",
"target_t_had_Pt", "target_t_had_Eta", "target_t_had_Phi", "target_t_had_E",
"target_t_lep_Pt", "target_t_lep_Eta", "target_t_lep_Phi", "target_t_lep_E",
]

output_columns_ptetaphiM = [
"target_W_had_Pt", "target_W_had_Eta", "target_W_had_Phi", "target_W_had_M",
"target_W_lep_Pt", "target_W_lep_Eta", "target_W_lep_Phi", "target_W_lep_M",
"target_b_had_Pt", "target_b_had_Eta", "target_b_had_Phi", "target_b_had_M",
"target_b_lep_Pt", "target_b_lep_Eta", "target_b_lep_Phi", "target_b_lep_M",
"target_t_had_Pt", "target_t_had_Eta", "target_t_had_Phi", "target_t_had_M",
"target_t_lep_Pt", "target_t_lep_Eta", "target_t_lep_Phi", "target_t_lep_M",
]

input_features_jets = [
"jet0 P_x",  "jet0 P_y",  "jet0 P_z",  "jet0 E",  "jet0 M",
"jet1 P_x",  "jet1 P_y",  "jet1 P_z",  "jet1 E",  "jet1 M",
"jet2 P_x",  "jet2 P_y",  "jet2 P_z",  "jet2 E",  "jet2 M",
"jet3 P_x",  "jet3 P_y",  "jet3 P_z",  "jet3 E",  "jet3 M",
"jet4 P_x",  "jet4 P_y",  "jet4 P_z",  "jet4 E",  "jet4 M",
]

input_features_jets_ptetaphi = [
"jet0 Pt",  "jet0 Eta",  "jet0 Phi",  "jet0 E", "jet0 M",
"jet1 Pt",  "jet1 Eta",  "jet1 Phi",  "jet1 E", "jet1 M",
"jet2 Pt",  "jet2 Eta",  "jet2 Phi",  "jet2 E", "jet2 M",
"jet3 Pt",  "jet3 Eta",  "jet3 Phi",  "jet3 E", "jet3 M",
"jet4 Pt",  "jet4 Eta",  "jet4 Phi",  "jet4 E", "jet4 M"
]

btags = [
"jet0 BTag", "jet1 BTag", "jet2 BTag", "jet3 BTag", "jet4 BTag"
]

input_features_lep = [
"lep Px", "lep Py", "lep Pz", "lep E", "met_met", "met_phi"
]

input_features_lep_ptetaphi = [
"lep Pt", "lep Eta", "lep Phi", "lep E", "met_met", "met_phi"
]

target_features_W_lep = [
"target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz", "target_W_lep_E"
]

target_features_W_had = [
"target_W_had_Px", "target_W_had_Py", "target_W_had_Pz", "target_W_had_E"
]

target_features_b_had  = [
"target_b_had_Px", "target_b_had_Py", "target_b_had_Pz", "target_b_had_E"
]

target_features_b_lep  = [
"target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz", "target_b_lep_E"
]

target_features_t_had  = [
"target_t_had_Px", "target_t_had_Py", "target_t_had_Pz", "target_t_had_E"
]

target_features_t_lep  = [
"target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz", "target_t_lep_E"
]

features_event_info = [
    "runNumber", "eventNumber", "weight", "jets_n", "bjets_n"
]

attributes = ['W_had_px', 'W_had_py', 'W_had_pz', 'W_had_E', 'W_had_m',
'W_had_pt', 'W_had_y', 'W_had_phi', 'b_had_px', 'b_had_py', 'b_had_pz',
'b_had_E', 'b_had_m', 'b_had_pt', 'b_had_y', 'b_had_phi', 't_had_px',
't_had_py', 't_had_pz', 't_had_E', 't_had_m', 't_had_pt', 't_had_y',
't_had_phi', 'W_lep_px', 'W_lep_py', 'W_lep_pz', 'W_lep_E', 'W_lep_m',
'W_lep_pt', 'W_lep_y', 'W_lep_phi', 'b_lep_px', 'b_lep_py', 'b_lep_pz',
'b_lep_E', 'b_lep_m', 'b_lep_pt', 'b_lep_y', 'b_lep_phi', 't_lep_px',
't_lep_py', 't_lep_pz', 't_lep_E', 't_lep_m', 't_lep_pt', 't_lep_y',
't_lep_phi']

corr_2d = [
'corr_W_had_pt', 'corr_W_had_y', 'corr_W_had_phi', 'corr_W_had_E',
'corr_b_had_pt', 'corr_b_had_y', 'corr_b_had_phi','corr_b_had_E',
'corr_t_had_pt','corr_t_had_y', 'corr_t_had_phi', 'corr_t_had_E',
'corr_W_lep_pt', 'corr_W_lep_y', 'corr_W_lep_phi', 'corr_W_lep_E',
'corr_b_lep_pt', 'corr_b_lep_y', 'corr_b_lep_phi','corr_b_lep_E',
 'corr_t_lep_pt','corr_t_lep_y', 'corr_t_lep_phi', 'corr_t_lep_E'
 ]

plots = ['reso_t_had_phi', 'corr_b_had_phi', 'corr_W_had_pt',
 'diff_W_lep_E', 'diff_t_had_y', 'reso_W_lep_pz',
 'diff_t_lep_m', 'reso_W_had_py', 'reso_t_lep_pt',
 'reso_W_had_px', 'b_had_phi', 'b_had_pt', 'diff_b_lep_E',
 'corr_W_lep_pt', 't_had_phi', 'reso_t_had_pt', 'diff_t_had_m',
 'reso_W_lep_py', 'diff_W_had_E', 'diff_t_lep_y', 'reso_W_had_pz',
 'corr_t_had_phi', 'reso_b_had_phi', 'reso_W_lep_px', 'diff_b_had_E',
 'b_lep_pt', 't_had_pt', 'W_lep_py', 'W_had_pz', 'corr_W_lep_phi',
 'reso_b_lep_pt', 'W_lep_px', 'W_lep_phi', 'W_lep_pz', 'W_had_py',
 't_lep_pt', 'reso_W_lep_phi', 'reso_b_had_pt', 'W_had_px',
 'reso_t_had_m', 'corr_b_had_E', 'reso_t_lep_y', 'reso_W_had_E',
 'corr_b_lep_phi', 'diff_W_lep_px', 'reso_t_lep_phi',
 'diff_W_lep_py', 'b_lep_phi', 'reso_b_had_E', 'corr_t_lep_y',
 'corr_W_had_E', 'diff_t_had_pt', 'diff_W_had_pz', 'corr_t_had_m',
 'corr_b_lep_E', 'reso_t_had_y', 'reso_W_lep_E', 'reso_t_lep_m',
 't_lep_phi', 'diff_W_had_px', 'corr_t_lep_m', 'diff_W_lep_pz',
 'diff_t_lep_pt', 'diff_W_had_py', 'reso_b_lep_E', 'corr_t_had_y',
 'corr_W_lep_E', 'reso_b_lep_phi', 'corr_t_lep_phi', 'corr_W_had_phi',
 'b_had_m', 'diff_b_had_pt', 'b_lep_y', 'W_had_phi', 't_had_E',
 'W_lep_y', 'W_had_m', 'b_had_y', 'diff_b_lep_pt', 'b_lep_m',
 'W_lep_m', 'reso_W_had_phi', 't_lep_E', 'W_had_y',
 'corr_W_lep_y', 'reso_b_lep_y', 'corr_t_had_E', 'diff_b_lep_px',
 'reso_b_had_m', 'diff_b_had_pz', 'diff_b_lep_py', 'reso_W_had_m',
 'reso_W_lep_y', 'corr_b_lep_y', 'reso_t_had_E', 'reso_b_lep_m',
 'corr_W_had_y', 'diff_b_had_px', 'reso_b_had_y', 'corr_t_lep_E',
 'diff_W_lep_phi', 'diff_b_had_py', 'diff_b_lep_pz', 'reso_W_had_y',
 'corr_b_had_y', 'reso_t_lep_E', 'reso_W_lep_m', 'W_had_E',
 't_lep_y', 't_had_m', 'diff_W_had_pt', 'diff_t_lep_py',
 'diff_t_had_pz', 'diff_t_lep_px', 'b_had_E', 'diff_t_had_phi',
 't_lep_m', 'W_lep_E', 't_had_y', 'diff_t_lep_pz', 'diff_b_had_phi',
 'diff_t_had_py', 'diff_W_lep_pt', 'b_lep_E', 'diff_t_had_px',
 'corr_b_had_pt', 'diff_b_lep_m', 't_lep_px', 'diff_b_had_y',
 'reso_b_had_py', 'reso_b_lep_pz', 'diff_t_lep_E', 'diff_W_had_y',
 'diff_W_lep_m', 'reso_b_had_px', 't_lep_py', 'W_had_pt', 't_had_pz',
 't_had_px', 'diff_W_had_phi', 'diff_b_lep_y', 'corr_b_lep_pt',
 'diff_b_had_m', 'reso_b_had_pz', 'reso_b_lep_py', 'reso_b_lep_px',
 'diff_W_had_m', 'diff_t_had_E', 'diff_W_lep_y', 't_lep_pz',
 'W_lep_pt', 't_had_py', 'b_had_pz', 'b_lep_py', 'reso_t_had_px',
 'reso_t_lep_pz', 'diff_t_lep_phi', 'reso_W_lep_pt',
 'reso_t_had_py', 'b_lep_px', 'corr_t_had_pt', 'b_had_py',
 'diff_b_lep_phi', 'b_lep_pz', 'reso_t_lep_px', 'reso_t_lep_py',
 'reso_W_had_pt', 'reso_t_had_pz', 'corr_t_lep_pt', 'b_had_px']
