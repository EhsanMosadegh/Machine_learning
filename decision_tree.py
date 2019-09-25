#!/usr/bin/python3

# Gini index function at a child node
def Gini_score( input_dataset , class_labels ) :
	# calculates Gini score for each split/input data set that includes class labels in it
	# input dat set can be parent data set or child data set

	total_rows_of_input_dataset = len( input_dataset )

	p_sqred_list = []

	for class_label in class_labels : # for each class label (binary=2)

		no_of_rows_with_class_i = input_dataset.count( class_label ) # how to count a param in a np.array column???

		probability_of_class_i = no_of_rows_with_class_i/total_rows_of_input_dataset

		p_sqred = probability_of_class_i**2

		p_sqred_list.append( p_sqred )

	Gini_for_input_dataset = 1-sum( p_sqred_list ) 

	return Gini_for_input_dataset , total_rows_of_input_dataset


###########################################################################################

class_labels = [ 'y' , 'n' ]

info_gain_all_features_list = []

# calculate information gain for each feature 
for feature in feature_list :  # feature_list = [x,y,z] == column labels

	total_rows_of_parent_dataset = len( np_array_collumn_of_feature )
	GiniScore_and_totalRows = Gini_score( input_np_array_feature_col ,  class_labels )
	Gini_score_of_parent_dataset = GiniScore_and_totalRows[0]


	# create a split dataset based on each featureValue
	# after we create splits in data set for each featureValue (e.g. male or female), we input it to the function
	# how create split in data set??? maybe filtering based on each featureValue???


	# split parent data set to child groups
	# split each feature to its featureValue groups

	feat_valu_group = [ 'male' , 'female'] 

	weight_of_each_group_list = []
	Gini_of_each_group_list = []

	# for each group/child node
	for feat_valu_split_t in feat_valu_group :  # ????

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








