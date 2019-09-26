#!/usr/bin/python3

import numpy as np

####################################################################################################

# Gini score function
# def Gini_score( feature_dataset , class_labels ) :
# 	# calculates Gini score for each split == input data set that includes class labels in it as the last column
# 	# input dat set can be parent data set or child data set, but the last column shold have classes

# 	total_rows_of_feature_dataset = len( feature_dataset )

# 	p_sqred_list = []

# 	for class_label in class_labels : # use class label to count the no of rows for each class label (binary=2)

# 		no_of_class_labels_with_class_i = feature_dataset[-1].count( class_label ) # how to count a param in a np.array column??? - should get the last column based on index

# 		probability_of_class_i = no_of_class_labels_with_class_i/total_rows_of_feature_dataset

# 		p_sqred = probability_of_class_i**2

# 		p_sqred_list.append( p_sqred )

# 	Gini_for_feature_dataset = 1-sum( p_sqred_list ) # sum of fraction of all classes

# 	return Gini_for_feature_dataset , total_rows_of_feature_dataset

####################################################################################################

def entropy( input_dataset , class_labels_list ) : # , class_labels ) :
	# calculates entropy based on each input dataset = Y = class labels in it as the last column

	total_rows_of_feature_dataset = len( input_dataset )

	p_log2p_list = []

	for class_label in class_labels_list : # use class label to count the no of rows for each class label (binary=2)
		print('######### H ######')
		print('-> input dataset in H=')
		print(input_dataset)

		classes_list = input_dataset[:,-1] #.tolist()
		print(f'-> class label= {class_label}')
		print(f'-> column of class labels= {classes_list}')
		#print(type( classes_list))

		classes_list = classes_list.tolist()
		no_of_class_labels = classes_list.count( class_label ) # get the last col==labels - how to count a param in a np.array 
		#del classes_list
		print( f'-> no. of Ci= {no_of_class_labels}')
		print(f'-> rows= {no_of_class_labels}')
		print(f'-> total rows= {total_rows_of_feature_dataset}')

		if (no_of_class_labels == 0 or total_rows_of_feature_dataset == 0) :
			p_log2p = 0
			print(f'-> we have an empty list... p_log2p=0')
			continue

		else:
			prob_of_class = no_of_class_labels/total_rows_of_feature_dataset
			print(f'-> prob= {prob_of_class}')
			p_log2p = prob_of_class * np.log2(prob_of_class)
			#print(f'-> plog2p= {p_log2p}')

			p_log2p_list.append( p_log2p )

	entropy_of_feature_dataset = -sum( p_log2p_list )
		#print('#########')

	return entropy_of_feature_dataset , total_rows_of_feature_dataset

####################################################################################################

# print(feature_dataset)
# print(" ")

#col_index = 2

def test_split_dataset( input_dataset , col_index , feature_value) :

	left = []
	right = []

	# define a filter here - for real values change the filter here
	# filter_left = ( input_dataset[ : , col_index ] == 0 )
	# filter_right = ( input_dataset[ : , col_index ] == 1 )
	for each_row in range(len( input_dataset[:,col_index] )) :

		row_value = input_dataset[ each_row , col_index ]
		print(f'-> row value is= {row_value}')
		print(f'-> featureValue is= {feature_value}')
		if row_value < feature_value :
			
			left.append( row_value )

		else:

			right.append( row_value )


	# left = input_dataset[ filter_left ]
	# right = input_dataset[ filter_right ]

	n_row_left = len( left )
	print( f'-> left is= {left}'  )

	n_row_right = len(right)
	print( f'-> right is= {right}'  )


	return left , right , n_row_left , n_row_right

####################################################################################################

# Select the best split point for a dataset
#def get_split( feature_dataset , label_dataset ) :
def get_split( input_dataset ) :


	# #label_dataset_trans = label_dataset[np.newaxis]
	# input_dataset = np.concatenate( (feature_dataset , label_dataset.T ) , axis=1 ) # join 2 lists vertically

	# get the class_labels
	class_labels_list = list( set(input_dataset[:,-1].flatten()) ) 
	print(f'-> class_labels_list is= {class_labels_list}')

	n_rows_parent = len( input_dataset[ :,0 ] ) # get the col based on col index

	print(f'-> no of feature columns = { (len( input_dataset[0,:]) -1) }')

	col_index_list = []
	featureValue_IG_list = []

	# loop over columns
	for col_index in range( (len( input_dataset[0,:]) -1) ) :
		for each_row in range(len(input_dataset[:,0])) :
			print( f'-> no of rows in column feature= {range(len(input_dataset[:,0]))} ')
			feature_value = input_dataset[ each_row , col_index ]
			print( f'-> Fvalue is= {feature_value}')

			# print( " ")
			# print(f'-> SPLIT for col {col_index}')
			#print( range(len(feature_dataset[0,:]) ) )
			#print( f'-> shape is = {feature_dataset.shape}' )
			#print(f'-> for col= {col_index}')
			#print(feature_dataset)

			entropy_parent = entropy( input_dataset , class_labels_list )
			# print(f'-> H-parent= {entropy_parent}')
			entropy_parent = entropy_parent[0]

			# print('-> input dataset before splitting=')
			# print(input_dataset)

			groups = test_split_dataset( input_dataset , col_index , feature_value ) #----> ????
			# print(';-> now groups that go into entropy are=')
			# print(groups)
			# print('-> left group=')
			# print(groups[0])

			entropy_left = entropy( groups[0] , class_labels_list )
			entropy_left = entropy_left[0]
			#print(type(entropy_left))
			# print( f'-> H left= {entropy_left}' )

		# print('-> right group=')
		# print(groups[1])
			entropy_right = entropy( groups[1] , class_labels_list )
			entropy_right = entropy_right[0]
			# print(f'-> H right= {entropy_right}')

			n_row_left = groups[2]
			n_row_right = groups[3]


			weight_left = n_row_left/n_rows_parent
			#print(type(weight_left))
			weight_right = n_row_right/n_rows_parent
			#print(type(weight_right))

			# get IG for each feature value
			left_contrib = (weight_left*entropy_left)
			print(f'-> left contrib= {left_contrib}')
			right_contrib = (weight_right*entropy_right)
			print(f'-> right contrib= {right_contrib}')

			# IG for each featureValue
			info_gain_featureValue = entropy_parent - ( left_contrib + right_contrib )
			print(f'-> for col={col_index}, IG is ={info_gain_featureValue} ')
			col_index_list.append( col_index )
			featureValue_IG_list.append( info_gain_featureValue )

	print( featureValue_IG_list)
	max_IG = max( featureValue_IG_list )
	print( f'-> max = {max_IG}')

	best_featureValue_index = featureValue_IG_list.index(max_IG)  #index(max( featureValue_IG_list ))
	print(best_featureValue_index )


	groups_left_right = test_split_dataset( input_dataset , col_index , max_IG )
	#print(f'-> groups= {groups_left_right}')
	print('1st group=')
	print(groups_left_right[0])
	print('2nd group=')
	print(groups_left_right[1])

	root_dict = {'index': best_featureValue_index , 'groups': groups_left_right }

	return root_dict

####################################################################################################

# Create a terminal node value
def to_terminal(group) :  # ????

	### array to list
	group = group.tolist()

	outcomes = [row[-1] for row in group]

	return max(set(outcomes), key=outcomes.count)

####################################################################################################

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth) :

	left, right = node['groups'][0:2]
	print('-> left, right lists=')
	print(left)
	print(right)

	del(node['groups'])

	# check if left or right is empty - for a no split
	print('-> type of left=')
	print(type(left))

	#if not left or not right:
	if ( left.size == 0 or right.size == 0 ) :
		print('-> we have an empty left-right array')
		node['left'] = node['right'] = to_terminal(left + right)  # ??? how add 2 arrays?
		return

	# check for max depth
	if depth >= max_depth:

		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return

	# process left child
	if len(left) <= min_size :
		node['left'] = to_terminal(left)

	else:

		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)

	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)

	else:

		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

####################################################################################################

# build a decision tree
def DT_train_binary(feature_dataset, label_dataset, max_depth ) :

	#label_dataset_trans = label_dataset[np.newaxis]
	input_dataset = np.concatenate( (feature_dataset , label_dataset.T ) , axis=1 ) # join 2 lists vertically
	min_size = 1 # the min number of samples in a node
	print('-> input dataset=')
	print(input_dataset)

	root_dict = get_split( input_dataset )

	split(root_dict , max_depth, min_size, 1)

	return root_dict



	# best_feature = get_best_split( feature_dataset , label_dataset )
	# print(" ")
	# print( f'the best feature= {best_feature}')


####################################################################################################

####################################################################################################

if __name__ == '__main__' :

	#feature_dataset_list = [ ['n','y','n'] , ['y','n','n'] , ['y','n','n'] , ['n','y','y'] , ['y','y','n'] , ['y','y','n'] ]
	#label_dataset_list = [ 'n' , 'n' , 'y' , 'y' , 'n' , 'y' ]
	# feature_dataset_list = [ [0,1] , [0,0] , [1,0] , [0,0] , [1,1] ]
	# #label_dataset_list = [ [1] , [0] , [0] , [0] , [1]]
	# label_dataset_list = [ 1,0,0,0,1]

	feature_dataset_list = [[2.771244718,1.784783929],
	[1.728571309,1.169761413],
	[3.678319846,2.81281357],
	[3.961043357,2.61995032],
	[2.999208922,2.209014212],
	[7.497545867,3.162953546],
	[9.00220326,3.339047188],
	[7.444542326,0.476683375],
	[10.12493903,3.234550982],
	[6.642287351,3.319983761]]

	label_dataset_list = [0,0,0,0,0,1,1,1,1,1]


	feature_dataset_arr = np.array( feature_dataset_list )
	label_dataset_arr = np.array( [label_dataset_list] ) # use [list] if we you want to transpose a list to column later
	max_depth = 1

	tree = DT_train_binary( feature_dataset_arr , label_dataset_arr , max_depth )
	print('-> final dictionary/tree is=')
	print(tree)


# ############################################################################################################