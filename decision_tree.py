#!/usr/bin/python3

# Gini index function at a child node
def Gini_score( input_dataset , class_labels ) :
	# calculates Gini score for each split == input data set that includes class labels in it as the last column
	# input dat set can be parent data set or child data set, but the last column shold have classes

	total_rows_of_input_dataset = len( input_dataset )

	p_sqred_list = []

	for class_label in class_labels : # use class label to count the no of rows for each class label (binary=2)

		no_of_rows_with_class_i = input_dataset[-1].count( class_label ) # how to count a param in a np.array column??? - should get the last column based on index

		probability_of_class_i = no_of_rows_with_class_i/total_rows_of_input_dataset

		p_sqred = probability_of_class_i**2

		p_sqred_list.append( p_sqred )

	Gini_for_input_dataset = 1-sum( p_sqred_list ) # sum of fraction of all classes

	return Gini_for_input_dataset , total_rows_of_input_dataset



def split_dataset( input_dataset[ feature_col_num??? ]  ) :


	# split any input data set and and return a subset of left and right

	return left_node_dataset , right_node_dataset

###########################################################################################

class_labels = [ 'y' , 'n' ]

info_gain_all_features_list = []

def info_gain_of_dataset( input_dataset , ) :

# calculate information gain for each feature/column 
for feature_col_num in len( input_dataset[:,-2] ) :  # should be col index, -2 the one before the last=class labels???

	total_rows_of_parent_dataset = len( input_dataset[ feature_col_num??? ] ) # get the col based on col index
	GiniScore_and_totalRows = Gini_score( input_dataset[ feature_col_num??? ] ,  class_labels )
	Gini_score_of_parent_dataset = GiniScore_and_totalRows[0]


	# create a split in dataset based on each featureValue
	# how create split in data set??? maybe filtering based on each featureValue???

	left_node_dataset , right_node_dataset = split_dataset( input_dataset[ feature_col_num??? ] )  # how split input dataset???


	# Gini score of each left and right nodes
	Gini_of_left_node , rows_of_left_node = Gini_score( left_node_dataset , class_labels )  
	Gini_of_right_node , rows_of_right_node = Gini_score( right_node_dataset , class_labels )  

	# a function to calculate IG for each feature
	IG_of_feature = IG_of_each_feature( Gini_score_of_parent_dataset , Gini_LEFT , GINI_RIGHT , n_row_left , n_row_right , n_row_total )













	# split parent data set to child groups
	# split each feature to its featureValue groups

	feat_valu_list = [ 'male' , 'female'] # update by hand???

	weight_of_each_group_list = []
	Gini_of_each_group_list = []

	###### now calculate IG for each feature

	# for each group/child node
	for feature_value in feat_valu_list :  # ????

		# calculate the Gini for each child split
		GiniScore_and_totalRows = Gini_score( split_dataset , class_labels )

		Gini_score_of_child_node = GiniScore_and_totalRows[0]
		Gini_of_each_group_list.append( Gini_score_of_child_node )

		total_rows_of_child_node = GiniScore_and_totalRows[1]
		weight_of_each_child_node = total_rows_of_child_node/total_rows_of_parent_dataset
		weight_of_each_group_list.append( weight_of_each_child_node )


	contribution_of_all_nodes_total_list = []

	# calculate parts of info gain equation for each child node
	for child_node in len( Gini_of_each_group_list ) :

		contribution_of_each_node = weight_of_each_group_list[ child_node ] * Gini_of_each_group_list[ child_node ]
		contribution_of_all_nodes_total_list.append( contribution_of_each_node )

	# now calculate info gain
	info_gain_for_a_feature = Gini_score_of_parent_dataset - sum( contribution_of_all_nodes_total_list )
	info_gain_all_features_list.append( info_gain_for_a_feature )








