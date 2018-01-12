import math, operator


def read_file(path):
	# read a file and make in usable dict format
	
	actual_class = {}
	data = {}
	
	with open(path, "r") as file:
		for i, line in enumerate(file):
			line_split = line.strip().split("\t")
			g_id = i
			actual_class[g_id] = int(line_split[-1])
			data[g_id] = line_split[:-1]
	return data, actual_class

	
def calculate_z_score(train_data):
	# calulate mean and standard deviation of columns of data
	
	count = len(train_data)
	
	sum = [0.0] * l
	for key, value in train_data.items():
		for i, v in enumerate(value):
			try:
				sum[i] = sum[i] + float(v)
			except:
				pass
	mean = [0.0] * l
	for i, s in enumerate(sum):
		mean[i] = s/count
	
	sum = [0.0] * l
	for key, value in train_data.items():
		for i, v in enumerate(value):
			try:
				sum[i] = sum[i] + (float(v) - mean[i])**2
			except:
				pass
	var = [1.0] * l
	for i, s in enumerate(sum):
		var[i] = math.sqrt(s/(count))
	
	return mean, var

	
def normalize_with_z_score(data, mean, std_dev):
	# normalize data using mean and standard deviation
	
	n_data = {}
	
	for key, value in data.items():
		n_value = [0.0] * l
		for i, v in enumerate(value):
			try:
				n_value[i] = (float(v) - mean[i])/(std_dev[i])
			except:
				n_value[i] = v
		n_data[key] = n_value
	return n_data


def normalize_data(train_data, test_data):
	# calcuate mean and standard deviation of train and normalize both
	
	mean, std_dev = calculate_z_score(train_data)
	n_train_data = normalize_with_z_score(train_data, mean, std_dev)
	n_test_data = normalize_with_z_score(test_data, mean, std_dev)
	
	return n_train_data, n_test_data


# In[34]:

def calculate_distance(test_value, value):
	# calculate euclidean distance between 2 records (also handling categorical data) 
	
	squared_sum = 0.0
	for t, v in zip(test_value, value):
		try:
			squared_sum = squared_sum + (float(t)-float(v))**2
		except:
			if t!=v:
				squared_sum = squared_sum + 1
	return (math.sqrt(squared_sum))


def classify_record(test_value, n_train_data, train_actual_class, k_val):
	# consider k nearest (most similar) records from train record for test record
	
	distance = {}
	for key, value in n_train_data.items():
		distance[key] = calculate_distance(test_value, value)
	kv_tuple = sorted(distance.items(), key=operator.itemgetter(1))
	
	class_dict = {}
	for key, value in kv_tuple[:k_val]:
		class_label = train_actual_class[key]
		if class_label not in class_dict:
			class_dict[class_label] = 0
		class_dict[class_label] = class_dict[class_label] + 1
	class_dict = sorted(class_dict.items(), key=operator.itemgetter(1))
	return class_dict[-1][0]


def classify_test_data(n_train_data, train_actual_class, n_test_data, k_val):
	# predict the values using knn logic
	
	test_knn_class = {}
	for key, value in n_test_data.items():
		test_knn_class[key] = classify_record(value, n_train_data, train_actual_class, k_val)
	return test_knn_class


def calculate_performance_measures(test_actual_class, test_knn_class, k_val):
	# comapare actual and predicted results, and calculate performance parameters
	
	a = 0.0
	b = 0.0
	c = 0.0
	d = 0.0
	
	for key, v in test_actual_class.items():
		# print(str(test_actual_class[key]) + "\t" + str(test_knn_class[key]))
		if test_actual_class[key] == 1:
			if test_knn_class[key] == 1:
				a = a + 1
			else:
				b = b + 1
		else:
			if test_knn_class[key] == 1:
				c = c + 1
			else:
				d = d + 1
	
	parameters = {"accuracy":100.0, "precision":100.0, "recall":100.0, "f_measure":100.0}
	try:
		parameters["accuracy"] = 100.0 * (a + d) / (a + b + c + d)
	except:
		pass
	try:
		parameters["precision"] = 100.0 * a / (a + c)
	except:
		pass
	try:
		parameters["recall"] = 100.0 * a / (a + b)
	except:
		pass
	try:
		parameters["f_measure"] = 100.0 * 2*a / (2*a + b + c)
	except:
		pass
	return parameters


def perform_knn(train_data, train_actual_class, test_data, test_actual_class, k_val):
	# normalize data, classify test data and calculate performance parameters
	
	n_train_data, n_test_data = normalize_data(train_data, test_data)
	test_knn_class = classify_test_data(n_train_data, train_actual_class, n_test_data, k_val)
	return calculate_performance_measures(test_actual_class, test_knn_class, k_val)


def split_dict(train_data, train_actual_class, n_fold_count):
	# split training data along with class, and create a list of n parts of training data along with class 
	
	train_data_list = []
	train_actual_class_list = []
	count = len(train_actual_class)
	l_split = math.floor(count/n_fold_count)
	
	for i  in range(0, n_fold_count):
		temp_data = {}
		temp_class = {}
		for j in range(i*l_split, (i+1)*l_split):
			temp_data[j] = train_data[j]
			temp_class[j] = train_actual_class[j]
		if(i+1==n_fold_count):
			for j in range((i+1)*l_split, count):
				temp_data[j] = train_data[j]
				temp_class[j] = train_actual_class[j]
		train_data_list = train_data_list + [temp_data]
		train_actual_class_list = train_actual_class_list + [temp_class]
	return train_data_list, train_actual_class_list


def knn_without_test(train_data, train_actual_class, n_fold_count, k_val):
	# knn without test data : knn with n-fold cross validation and calculate average performance parameters of all folds
	
	train_data_list, train_actual_class_list = split_dict(train_data, train_actual_class, n_fold_count)
	parameters = {"accuracy":0.0, "precision":0.0, "recall":0.0, "f_measure":0.0}
	
	for i in range(n_fold_count):
		train_data = {}
		train_actual_class = {}
		test_data = train_data_list[i]
		test_actual_class = train_actual_class_list[i]

		for j in range(n_fold_count):
			if i==j:
				continue
			train_data.update(train_data_list[j])
			train_actual_class.update(train_actual_class_list[j])
		new_parameters = perform_knn(train_data, train_actual_class, test_data, test_actual_class, k_val)
		for key, new_value in new_parameters.items():
			parameters[key] = parameters[key] + new_value
	for key, new_value in new_parameters.items():
		parameters[key] = parameters[key]/10
	return parameters


def knn_with_test(train_data, train_actual_class, test_path, k_val):
	# knn with test data, and calculate performance parameters
	
	test_data, test_actual_class = read_file(test_path)
	return perform_knn(train_data, train_actual_class, test_data, test_actual_class, k_val)


# train_path = "project3_dataset1.txt" or "project3_dataset2.txt" or "project3_dataset3_train.txt"
train_path = input("Enter training file path: ")

train_data, train_actual_class = read_file(train_path)
l = len(train_data[0])

k_val = int(input("Enter value of k: "))

# test_data_flag = 0 or 1
test_data_flag = input("Do we have test data (0 for NO): ")
test_data_flag = int(test_data_flag)

if test_data_flag:
	# test_path = "project3_dataset3_test.txt"
	test_path = input("Enter testing file path: ")
	print(k_val, knn_with_test(train_data, train_actual_class, test_path, k_val))
else:
	# n_fold_count = 10
	n_fold_count = int(input("Enter value of n for n-fold: "))
	print(k_val, knn_without_test(train_data, train_actual_class, n_fold_count, k_val))

