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
		# print('######### H ######')
		# print('-> input dataset in H=')
		# print(input_dataset)

		classes_list = input_dataset[:,-1] #.tolist()
		# print(f'-> class label= {class_label}')
		# print(f'-> column of class labels= {classes_list}')
		#print(type( classes_list))

		classes_list = classes_list.tolist()
		no_of_class_labels = classes_list.count( class_label ) # get the last col==labels - how to count a param in a np.array 
		#del classes_list
		# print( f'-> no. of Ci= {no_of_class_labels}')
		# print(f'-> rows= {no_of_class_labels}')
		# print(f'-> total rows= {total_rows_of_feature_dataset}')

		if (no_of_class_labels == 0 or total_rows_of_feature_dataset == 0) :
			p_log2p = 0
			#print(f'-> we have an empty list... p_log2p=0')
			continue

		else:
			prob_of_class = no_of_class_labels/total_rows_of_feature_dataset
			#print(f'-> prob= {prob_of_class}')
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
		# print(f'-> row value is= {row_value}')
		# print(f'-> featureValue is= {feature_value}')
		if row_value < feature_value :
			
			left.append( row_value )

		else:

			right.append( row_value )


	# left = input_dataset[ filter_left ]
	# right = input_dataset[ filter_right ]

	n_row_left = len( left )
	# print( f'-> left is= {left}'  )

	n_row_right = len(right)
	# print( f'-> right is= {right}'  )


	return left , right , n_row_left , n_row_right

####################################################################################################

# Select the best split point for a dataset
#def get_split( feature_dataset , label_dataset ) :
def get_split( input_dataset ) :


	# #label_dataset_trans = label_dataset[np.newaxis]
	# input_dataset = np.concatenate( (feature_dataset , label_dataset.T ) , axis=1 ) # join 2 lists vertically

	# get the class_labels
	class_labels_list = list( set(input_dataset[:,-1].flatten()) ) 
	# print(f'-> class_labels_list is= {class_labels_list}')

	n_rows_parent = len( input_dataset[ :,0 ] ) # get the col based on col index

	# print(f'-> no of feature columns = { (len( input_dataset[0,:]) -1) }')

	col_index_list = []
	featureValue_IG_list = []

	# loop over columns
	for col_index in range( (len( input_dataset[0,:]) -1) ) :
		for each_row in range(len(input_dataset[:,0])) :
			# print( f'-> no of rows in column feature= {range(len(input_dataset[:,0]))} ')
			feature_value = input_dataset[ each_row , col_index ]
			# print( f'-> Fvalue is= {feature_value}')

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
			# print(f'-> left contrib= {left_contrib}')
			right_contrib = (weight_right*entropy_right)
			#print(f'-> right contrib= {right_contrib}')

			# IG for each featureValue
			info_gain_featureValue = entropy_parent - ( left_contrib + right_contrib )
			#print(f'-> for col={col_index}, IG is ={info_gain_featureValue} ')
			col_index_list.append( col_index )
			featureValue_IG_list.append( info_gain_featureValue )

	#print( featureValue_IG_list)
	max_IG = max( featureValue_IG_list )
	#print( f'-> max = {max_IG}')

	best_featureValue_index = featureValue_IG_list.index(max_IG)  #index(max( featureValue_IG_list ))
	#print(best_featureValue_index )


	groups_left_right = test_split_dataset( input_dataset , col_index , max_IG )
	#print(f'-> groups= {groups_left_right}')
	# print('1st group=')
	# print(groups_left_right[0])
	# print('2nd group=')
	# print(groups_left_right[1])

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
	# print('-> input dataset=')
	# print(input_dataset)

	root_dict = get_split( input_dataset )

	split(root_dict , max_depth, min_size, 1)

	return root_dict



	# best_feature = get_best_split( feature_dataset , label_dataset )
	# print(" ")
	# print( f'the best feature= {best_feature}')


####################################################################################################

####################################################################################################

# if __name__ == '__main__' :

# 	#feature_dataset_list = [ ['n','y','n'] , ['y','n','n'] , ['y','n','n'] , ['n','y','y'] , ['y','y','n'] , ['y','y','n'] ]
# 	#label_dataset_list = [ 'n' , 'n' , 'y' , 'y' , 'n' , 'y' ]
# 	# feature_dataset_list = [ [0,1] , [0,0] , [1,0] , [0,0] , [1,1] ]
# 	# #label_dataset_list = [ [1] , [0] , [0] , [0] , [1]]
# 	# label_dataset_list = [ 1,0,0,0,1]

# 	feature_dataset_list = [[2.771244718,1.784783929],
# 	[1.728571309,1.169761413],
# 	[3.678319846,2.81281357],
# 	[3.961043357,2.61995032],
# 	[2.999208922,2.209014212],
# 	[7.497545867,3.162953546],
# 	[9.00220326,3.339047188],
# 	[7.444542326,0.476683375],
# 	[10.12493903,3.234550982],
# 	[6.642287351,3.319983761]]

# 	label_dataset_list = [0,0,0,0,0,1,1,1,1,1]


# 	feature_dataset_arr = np.array( feature_dataset_list )
# 	label_dataset_arr = np.array( [label_dataset_list] ) # use [list] if we you want to transpose a list to column later
# 	max_depth = 1

# 	tree = DT_train_binary( feature_dataset_arr , label_dataset_arr , max_depth )
# 	# print('-> final dictionary/tree is=')
# 	# print(tree)


# ############################################################################################################
import numpy as np
from math import log
def calc_entropy(Y):
	elems, counts = np.unique(Y, return_counts= True)
	entropy = 0.0
	for element in range(len(elems)):
		p_x = counts[element]/np.sum(counts)
		entropy -= p_x * log(p_x, 2)
	return entropy
def mode(arr):
	values, counts = np.unique(arr, return_counts=True)
	ind = np.argmax(counts)
	return values[ind]
def ID3(samples, originalSample, originalLabel,label, headers, max_depth, depth):
	#Stopping conditions
	if depth == max_depth:
		return mode(originalLabel)
	if np.size(samples)== 0:
		return mode(originalLabel)
	if len(np.unique(label)) ==1:
		return mode(label)
	num_features = np.size(samples,1)
	IG=[]
	for feat in range(num_features):
		IG.append(calc_attr_entropy(samples[:,feat], label))
	best_feature_index = np.argmax(IG)
	tree = {headers[best_feature_index]:{}}


	element = headers[best_feature_index]
	index =headers.index(element)
	headers.pop(index)




	leftSub, leftLabel, rightSub, rightLabel= split(samples, label, best_feature_index)
	if(np.size(leftSub)==0):
		tree[element][0] = mode(samples[:,best_feature_index])
	else:

		subTree1 = ID3(leftSub, originalSample, originalLabel,leftLabel, headers, max_depth, depth+1)
		tree[element][0] = subTree1
	if(np.size(rightSub)==0):
		tree[element][1] = mode(samples[:,best_feature_index])
	else:

		subTree2 = ID3(rightSub, originalSample, originalLabel,rightLabel,headers, max_depth, depth+1)
		tree[element][1] = subTree2

	return(tree)


def calc_attr_entropy(column, label):
	
	zero_rows_index = np.where(column==0) #array of indicies of feature column with a value of 1
	one_rows_index = np.where(column==1)  #array of indicies of feature column with a value of 0
	
	len1 = np.size(one_rows_index) #length of the index array
	len0 = np.size(zero_rows_index)
	
	totalLen = len1+len0
	if totalLen == 0:
		return 0
	
	trueTrue =trueFalse = 0     #counts of feature values that are equal to 1
	falseFalse = falseTrue =  0 #counts of feature values that are equal to 0
	if len0 == 0:
		p00=0
		p01=0
	else:
		for i in np.nditer(zero_rows_index):
			if label[i] == 1:
				falseTrue +=1
			else:
				falseFalse +=1
	if len1 == 0:
		p11 =0
		p10 =0
	else:
		for i in np.nditer(one_rows_index):
			if label[i] == 1:
				trueTrue +=1
			else:
				trueFalse +=1
	

	if len1 == 0:
		p_1_1 = 0
		p_1_0 = 0
	else:
		p_1_1 = trueTrue/len1
		p_1_0 = trueFalse/len1

	if len0 == 0:
		p_0_0 = 0
		p_0_1= 0
	else:
		p_0_0 = falseFalse/len0
		p_0_1 = falseTrue/len0

	if p_0_0 > 0:
		p00 = p_0_0*log(p_0_0 , 2)
	else:
		p00 = 0
	if p_0_1 > 0:
		p01 = p_0_1*log(p_0_1 , 2)
	else:
		p01 = 0
	
	if p_1_0 > 0:
		p10 = p_1_0*log(p_1_0 , 2)
	else:
		p10 = 0
	if p_1_1 > 0:
		p11 = p_1_1*log(p_1_1 , 2)
	else:
		p11 = 0


	a1 = len1/totalLen * -((p11+p10))
	a0 = len0/totalLen * -(p00+p01)
	return calc_entropy(label) - (a1+a0)
def DT_train_binary(X, Y, max_depth):
	headers= [i for i in range(np.size(X,1))]
	tree = ID3(X, X, Y,Y, headers, max_depth, depth=0)
	calc_entropy(Y)
	
	return 	tree


def split(X, Y, best_feature_index):
	leftSubset = []
	leftLabel = []
	rightSubset = []
	rightLabel = []


	for feat in range(np.size(Y,0)):
		if X[feat,best_feature_index] == 0:
			arr = X[feat, :]
			leftSubset.append(arr)
			leftLabel.append(Y[feat])
		elif X[feat, best_feature_index]==1:
			arr = X[feat, :]
			rightSubset.append(arr)
			rightLabel.append(Y[feat])
	
	leftNewData = np.array(leftSubset)
	leftNewLabel = np.array(leftLabel)
	rightNewData = np.array(rightSubset)
	rightNewLabel= np.array(rightLabel)

	if np.size(leftNewData) != 0:
		leftNewData = np.delete(leftNewData, best_feature_index, axis=1)
	if np.size(rightNewData) != 0:
		rightNewData = np.delete(rightNewData, best_feature_index, axis=1)
	return leftNewData, leftNewLabel, rightNewData, rightNewLabel
def DT_make_prediction(x,DT):
	for i in DT.keys():
		branch = x[i]
		DT = DT[i][branch]
		if type(DT) is dict:
			prediction = DT_make_prediction(x, DT)
		else:
			prediction = DT
			break;                            
	return prediction
def DT_test_binary(X,Y, DT):
	predictionList = []
	
	for row in range(np.size(X, 0)):
		predictionList.append(DT_make_prediction(X[row, :], DT))
	
	predictionArr = np.array(predictionList)
	predictionArr = np.reshape(predictionArr, (np.size(X,0),1))
	correct = 0
	
	for row in range(np.size(X,0)):
		if predictionArr[row] == Y[row]:
			correct+=1
	return (correct /np.size(X,0))*100
def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
	dtList= []
	#Make a DT for every possible 
	dtList.append(DT_train_binary(X_train,Y_train,-1))
	for i in range(1,np.size(Y_train,0)):
		dtList.append(DT_train_binary(X_train,Y_train,i))
	accuracyList =[]
	for i in range(len(dtList)):
		accuracyList.append(DT_test_binary(X_val,Y_val, dtList[i]))
	bestTreeIndex = np.argmax(accuracyList)
	return 	dtList[bestTreeIndex]