import os
import pandas as pd
import scipy.sparse as sp
import scipy.io as spio

def dump(data_dict, full_file_path) :
	file_path, file_name_prefix = '/'.join(full_file_path.split('/')[:-1]), full_file_path.split('/')[-1]

	if not os.path.isdir(file_path):
	    os.makedirs(file_path)


	with open(file_path + '/' + file_name_prefix + "_fileindex.txt", 'wt') as f :
		for file_name_suffix in data_dict :
			data = data_dict[file_name_suffix]

			file_name = ''
			file_type = ''

			if isinstance(data, pd.DataFrame) :
				file_name = file_name_prefix + "_" + file_name_suffix + ".csv"
				file_type = 'csv'

				data.to_csv(file_path + '/' + file_name, sep='\t', index=False)
			elif isinstance(data, sp.csr_matrix) or isinstance(data, sp.csc_matrix) or isinstance(data, sp.coo_matrix) or isinstance(data, sp.lil_matrix) :
				file_name = file_name_prefix + "_" + file_name_suffix + ".mat"
				file_type = 'mat'

				spio.savemat(file_path + '/' + file_name, {'data_mat' : data})

			f.write(file_name + '\t' + file_name_suffix + '\t' + file_type + '\n')

def load(full_file_path) :
	file_path, file_name_prefix = '/'.join(full_file_path.split('/')[:-1]), full_file_path.split('/')[-1]

	data_dict = {}
	with open(file_path + '/' + file_name_prefix + "_fileindex.txt", 'rt') as f :
		for line in f.readlines() :
			file_name, file_name_suffix, file_type = line.strip().split('\t')

			if file_type == 'csv' :
				data = pd.read_csv(file_path + '/' + file_name, sep='\t')
			elif file_type == 'csv' :
				data = spio.loadmat(file_path + '/' + file_name)['data_mat']

			data_dict[file_name_suffix] = data

	return data_dict
