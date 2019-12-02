#!/usr/bin/python3

import numpy as np

def choose_K(X_train, Y_train, X_val, Y_val):

	best_K_list = []

	for val_sample in range(X_val.shape[0]):

		x_val = X_val[val_sample, :]
		y_val = Y_val[val_sample, :]

		# print(f'-> x val= {x_val}')
		dist_list = []

		for train_sample in range(X_train.shape[0]):

			x_train = X_train[train_sample, :]

			diff_elements = np.subtract(x_train, x_val)
			squared_diff = np.square(diff_elements)
			dist = (np.sum(squared_diff))**0.5
			dist_list.append(dist)

		# print(f'-> dist list= {dist_list}')

		Y_train_transposed = np.transpose(Y_train)
		# print( Y_train_transposed)
		Y_train_tolist = Y_train_transposed.tolist()[0]
		# print( np.array(Y_train_tolist).shape )
		# print(Y_train_tolist)
		dist_array = [dist_list, Y_train_tolist]
		#print(f'-> dist array= {np.array(dist_array).shape }')

		dist_array = np.transpose(np.array(dist_array))
		#print(dist_array)

		# sort based on dist col = the 1st column = 0
		sorted_dist_array = dist_array[dist_array[:, 0].argsort()]
		# print('-> sorted')
		#print(sorted_dist_array)

		accuracy_list = []
		K_list = []

		for K in range(1, sorted_dist_array.shape[0]+1):
			#print(f'-> K= {K}')
			k_near_labels = sorted_dist_array[0:K, 1]
			#print(f'-> k nears= {k_near_labels}')
			correct_labels = (k_near_labels.tolist()).count(y_val)
			#print(f'-> y_val= {y_val} and correct labels= {correct_labels}')
			accuracy = correct_labels/K
			accuracy_list.append(accuracy)
			K_list.append(K)

		# print(accuracy_list)

		val_max_acc = np.amax(accuracy_list)
		# print(f'-> max= {val_max_acc}')
		k_max_index = np.where(accuracy_list==val_max_acc)
		# print(f'-> K max index list= {k_max_index}')
		# we get the first index/element of a tuple as the first K for max accuracy
		indx = (k_max_index[0])[0]
		# print(f'-> index of the first max K= {indx} ')
		K_best = K_list[indx]
		# print(K_best)
		best_K_list.append(K_best)
	
	return best_K_list



# if __name__ == '__main__':

# 	X1 = [[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]]
# 	Y1 = [[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]]

# 	X2 = [[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]]
# 	Y2 = [[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]]



# 	X_train = np.array(X1)
# 	Y_train = np.array(Y1)

# 	X_val = np.array(X2)
# 	Y_val = np.array(Y2)

# 	best_K_list = choose_K(X_train, Y_train, X_val, Y_val)

# 	print(f'-> no of val data= {len(Y_val)}')
# 	print(f'-> no of best K= {len(best_K_list)}')
# 	print(best_K_list)
