#!/usr/bin/python3

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



import numpy as np

def entropy( input_dataset , class_labels_list ) : # , class_labels ) :
	# calculates entropy based on each input dataset = Y = class labels in it as the last column

	total_rows_of_feature_dataset = len( input_dataset )

	p_log2p_list = []

	for class_label in class_labels_list : # use class label to count the no of rows for each class label (binary=2)
		print('#########')
		classes_list = input_dataset[:,-1].tolist()
		print(f'-> class label= {class_label}')
		print(f'-> classes list= {classes_list}')

		no_of_class_labels = classes_list.count( class_label ) # get the last col==labels - how to count a param in a np.array column??? - should get the last column based on index
		#print( f'-> no. of Ci= {no_of_class_labels}')
		#print(f'-> rows= {no_of_class_labels}')
		#print(f'-> total rows= {total_rows_of_feature_dataset}')

		prob_of_class = no_of_class_labels/total_rows_of_feature_dataset
		#print(f'-> prob= {prob_of_class}')

		if (prob_of_class == 0 ) :
			p_log2p = 0
			#print(f'-> prob is= 0, we get the p_log2p= 0')

		else:
			p_log2p = prob_of_class * np.log2(prob_of_class)
			#print(f'-> plog2p= {p_log2p}')

		p_log2p_list.append( p_log2p )

		entropy_of_feature_dataset = -sum( p_log2p_list )
		#print('#########')

	return entropy_of_feature_dataset , total_rows_of_feature_dataset



# print(feature_dataset)
# print(" ")

#col_index = 2

def split_dataset( input_dataset , col_index ) :

	# define a filter here - for real values change the filter here
	filter_left = ( input_dataset[ : , col_index ] == 0 )
	filter_right = ( input_dataset[ : , col_index ] == 1 )

	left = input_dataset[ filter_left ]
	right = input_dataset[ filter_right ]

	n_row_left = len(left[:,0])
	n_row_right = len(right[:,0])

	return left , right , n_row_left , n_row_right




# left_list = split_dataset( feature_dataset , col_index )[0]
# right_list = split_dataset( feature_dataset , col_index )[1]


# print(left_list)
# print(right_list)


# Select the best split point for a dataset
def get_best_split( feature_dataset , label_dataset ) :

	#label_dataset_trans = label_dataset[np.newaxis]
	input_dataset = np.concatenate( (feature_dataset , label_dataset.T ) , axis=1 ) # join 2 lists vertically

	# get the class_labels
	class_labels_list = list( set(label_dataset.flatten() ) )

	n_rows_parent = len( feature_dataset[ :,0 ] ) # get the col based on col index

	col_index_list = []
	IG_feature_list = []

	# loop over columns
	for col_index in range(len(feature_dataset[0,:]) ) :
		print(f'-> for col {col_index}')
		#print( range(len(feature_dataset[0,:]) ) )
		#print( f'-> shape is = {feature_dataset.shape}' )
		#print(f'-> for col= {col_index}')
		#print(feature_dataset)

		entropy_parent = entropy( input_dataset , class_labels_list )
		print(f'-> H-parent= {entropy_parent}')
		entropy_parent = entropy_parent[0]


		groups = split_dataset( input_dataset , col_index )

		entropy_left = entropy( groups[0] , class_labels_list )
		entropy_left = entropy_left[0]
		#print(type(entropy_left))
		print( f'-> H left= {entropy_left}' )

		entropy_right = entropy( groups[1] , class_labels_list )
		entropy_right = entropy_right[0]
		print(f'-> H right= {entropy_right}')

		n_row_left = groups[2]
		n_row_right = groups[3]



		weight_left = n_row_left/n_rows_parent
		#print(type(weight_left))
		weight_right = n_row_right/n_rows_parent
		#print(type(weight_right))

		# get IG for each feature
		left_contrib = (weight_left*entropy_left)
		print(f'-> left contrib= {left_contrib}')
		right_contrib = (weight_right*entropy_right)
		print(f'-> right contrib= {right_contrib}')

		info_gain_feature = entropy_parent - ( left_contrib + right_contrib )
		print(f'-> for col={col_index}, IG is ={info_gain_feature} ')
		col_index_list.append( col_index )
		IG_feature_list.append( info_gain_feature )

	print( IG_feature_list)
	max_IG = max( IG_feature_list )
	print( f'-> max = {max_IG}')
	best_feature_index = IG_feature_list.index(max_IG)  #index(max( IG_feature_list ))
	print(best_feature_index )

	groups_left_right = split_dataset( feature_dataset , best_feature_index )
	#print(f'-> groups= {groups_left_right}')
	print('1st group=')
	print(groups_left_right[0])
	print('2nd group=')
	print(groups_left_right[1])

	best_feature_dict = {'index': best_feature_index , 'groups': groups_left_right }

	return best_feature_dict


def DT_train_binary(feature_dataset,label_dataset) :

	best_feature = get_best_split( feature_dataset , label_dataset )
	print(" ")
	print( f'the best feature= {best_feature}')


####################################################################################################

if __name__ == '__main__' :

	#feature_dataset_list = [ ['n','y','n'] , ['y','n','n'] , ['y','n','n'] , ['n','y','y'] , ['y','y','n'] , ['y','y','n'] ]
	#label_dataset_list = [ 'n' , 'n' , 'y' , 'y' , 'n' , 'y' ]
	feature_dataset_list = [ [0,1] , [0,0] , [1,0] , [0,0] , [1,1] ]
	#label_dataset_list = [ [1] , [0] , [0] , [0] , [1]]
	label_dataset_list = [ 1,0,0,0,1]


	feature_dataset_arr = np.array( feature_dataset_list )
	label_dataset_arr = np.array( [label_dataset_list] ) # use [list] if we you want to transpose a list to column later
	
	DT_train_binary( feature_dataset_arr , label_dataset_arr )









# ############################################################################################################


# def split_dataset( feature_dataset[ feature_col_num??? ]  ) :

# 	# why we should get 2 lists of left and right ?



# 	# split any input data set and and return a subset of left and right

# 	return left_node_dataset , right_node_dataset


# def IG_of_each_feature( ) :

# 	weight_left = n_row_left/n_row_total
# 	weight_right = n_row_right/n_row_total

# 	contribut_left = weight_left*Gini_left
# 	contribut_right = weight_right* Gini_right

# 	info_gain_of_feature = Gini_score_of_parent_dataset - sum( contribut_left , contribut_right )

# 	return info_gain_of_feature



# def get_split( feature_dataset , ) :

# 	class_labels = [ 'y' , 'n' ]  # should get automatically -- list(set(row[-1] for row in dataset))

# 	info_gain_of_all_features_list = []

# 	# calculate information gain for each feature/column
# 	for feature_col_num in len( feature_dataset[:,-2] ) :  # should be col index, -2 the one before the last=class labels???

# 		total_rows_of_parent_dataset = len( feature_dataset[ feature_col_num??? ] ) # get the col based on col index
# 		GiniScore_and_totalRows = Gini_score( feature_dataset[ feature_col_num??? ] ,  class_labels )
# 		Gini_score_of_parent_dataset = GiniScore_and_totalRows[0]


# 		# create a split in dataset based on each featureValue
# 		# how create split in data set??? maybe filtering based on each featureValue???

# 		left_node_dataset , right_node_dataset = split_dataset( feature_dataset[ feature_col_num??? ] )  # how split input dataset???


# 		# Gini score of each left and right nodes
# 		Gini_of_left_node , rows_of_left_node = Gini_score( left_node_dataset , class_labels )
# 		Gini_of_right_node , rows_of_right_node = Gini_score( right_node_dataset , class_labels )

# 		# a function to calculate IG for each feature
# 		IG_of_feature = IG_of_each_feature( Gini_score_of_parent_dataset , Gini_left , Gini_right , n_row_left , n_row_right , n_row_total )

# 		info_gain_of_all_features_list.append( IG_of_feature )


# 	# this is the node in our DT
# 	index_of_best_split = max( info_gain_of_all_features_list )  # how get the index of feature????


# 	feat_index , best_split_IG = index_of_best_split , info_gain_of_all_features_list[ index_of_best_split ]

# 	split_info_dict = { 'col_index' : feat_index , 'best_IG' : best_split_IG }

# 	return split_info_dict






# ################## the main function ################

# sample_dataset= [ ]


# split = get_split( feature_dataset )



















#def dt_train_binary( inputs ) :






























	# # split parent data set to child groups
	# # split each feature to its featureValue groups

	# feat_valu_list = [ 'male' , 'female'] # update by hand???

	# weight_of_each_group_list = []
	# Gini_of_each_group_list = []

	# ###### now calculate IG for each feature

	# # for each group/child node
	# for feature_value in feat_valu_list :  # ????

	# 	# calculate the Gini for each child split
	# 	GiniScore_and_totalRows = Gini_score( split_dataset , class_labels )

	# 	Gini_score_of_child_node = GiniScore_and_totalRows[0]
	# 	Gini_of_each_group_list.append( Gini_score_of_child_node )

	# 	total_rows_of_child_node = GiniScore_and_totalRows[1]
	# 	weight_of_each_child_node = total_rows_of_child_node/total_rows_of_parent_dataset
	# 	weight_of_each_group_list.append( weight_of_each_child_node )


	# contribution_of_all_nodes_total_list = []

	# # calculate parts of info gain equation for each child node
	# for child_node in len( Gini_of_each_group_list ) :

	# 	contribution_of_each_node = weight_of_each_group_list[ child_node ] * Gini_of_each_group_list[ child_node ]
	# 	contribution_of_all_nodes_total_list.append( contribution_of_each_node )

	# # now calculate info gain
	# info_gain_for_a_feature = Gini_score_of_parent_dataset - sum( contribution_of_all_nodes_total_list )
	# info_gain_of_all_features_list.append( info_gain_for_a_feature )








