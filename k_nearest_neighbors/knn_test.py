#!/usr/bin/python3

import numpy as np # todo import numpy and pandas later

def KNN_test(X_train, Y_train, X_test, Y_test, K):

	test_sample_list = []
	accuracy_list = []
	# print(f'-> len Y test= {len( Y_test )}')

	for test_sample in range(X_test.shape[0]):

		# print(f'-> test sameple= {test_sample}')

		x_test_sample = X_test[test_sample, :]
		y_test_label = Y_test[test_sample, :]

		# print(f'-> x test= {x_test_sample}')
		# print(f'-> y test= {y_test_label}')

		dist_list = []

		for train_sample in range(X_train.shape[0]):

			# print(f'-> train row range= {range(X_train.shape[0])}')

			x_train_sample = X_train[train_sample, :]
			# print(f'-> x train ndim= { x_train_sample }')

			# calculate distance
			squesred_diff_list = []

			for element in range(x_train_sample.size):
				# print(f'-> element= {element}')
				# print(f'-> range= {range(x_train_sample.size)}')

				diff_of_elements_powered2 = (x_train_sample[element] - x_test_sample[element])**2
				squesred_diff_list.append(diff_of_elements_powered2)

			dist = (sum(squesred_diff_list))**(0.5)
			dist_list.append(dist)

		# print(f'-> dist list len= {len(dist_list)}')
		# print(f'-> dist list type= {type(dist_list)}')

		# print(f'-> Y train type= {type(Y_train)}')
		# print(f'-> Y train shape= {Y_train.shape}')
		Y_train_transposed = np.transpose(Y_train)
		# print( Y_train_transposed)
		Y_train_tolist = Y_train_transposed.tolist()[0]
		# print( np.array(Y_train_tolist).shape )
		# print(Y_train_tolist)
		dist_array = [dist_list, Y_train_tolist]
		# print(dist_array)
		# print(np.array(dist_array).shape)
		dist_array = np.transpose(np.array(dist_array))
		# print(dist_array)
		# print(" ")
		# sort based on the 1st column = 0
		sorted_dist_array = dist_array[ dist_array[:,0].argsort() ]
		# print(sorted_dist_array)

		# select K nearest; K is not zero here
		k_near_labels = sorted_dist_array[0:K, 1]
		# print(f'-> K near class labels= {k_near_labels}')
		test_label = y_test_label
		# print(f'-> test label= {test_label}')
		correct_labels = (k_near_labels.tolist()).count(test_label)
		# print(f'-> correct pred= {correct_labels}')

		accuracy = (correct_labels/K)

		accuracy_list.append(accuracy)
		test_sample_list.append(test_sample)

	# print( f'-> for test samples= {test_sample_list}')
	# print( f'-> accuracy is= {accuracy_list}')

	return accuracy_list



if __name__ == '__main__':

	X1 = [[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]]
	Y1 = [[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]]

	X2 = [[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]]
	Y2 = [[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]]

	X_train = np.array(X1)
	Y_train = np.array(Y1)

	X_test = np.array(X2)
	Y_test = np.array(Y2)

	K = 5

	accuracy_list = KNN_test(X_train, Y_train, X_test, Y_test, K)

	print(f'-> number of neighboring points= K= {K}')
	# print(f'-> list of test data points that were checked= {test_sample_list}')
	print(f'-> accuracy of test data= {accuracy_list}')






