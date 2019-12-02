#!/usr/bin/python3

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
