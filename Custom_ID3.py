import numpy as np
import pandas as pd
import Phrase_WuPalmer as phrase
import sys
#sys.setrecursionlimit(1900) # Uncomment for the program to handle larger datasets
###################################################################################################################################################################################################################

# Calculates the entropy of the dependant variable in a given dataframe
def entropy(df):
    dependant_variable = df.keys()[-1] # Assign the last key of the dataframe as the dependant variable
    entropy = 0
    unique_values = df[dependant_variable].unique() # Get the unique values of the dependant variable
    
    for value in unique_values:
        probability = df[dependant_variable].value_counts()[value]/len(df[dependant_variable])
        entropy += -probability*np.log2(probability)
        
    return entropy


# Calculates the entropy of an attribute in a given dataframe
def find_entropy_attribute(df, attribute):
    target_class = df.keys()[-1] 
    target_class_values = df[target_class].unique() 

    if type(df[attribute][0]) == str:
        # Find unique attribute values for non-numeric attributes
        attribute_values = df[attribute].unique()  
        overall_entropy = 0

        for attribute_value in attribute_values:
            entropy = 0

            # Calculate entropy for each attribute value
            for target_class_value in target_class_values:
                numerator = len(df[(df[attribute] == attribute_value) & (df[target_class] == target_class_value)])
                denominator = len(df[df[attribute] == attribute_value])
                probability = numerator / (denominator + np.finfo('float').eps)
                entropy += -probability * np.log2(probability + np.finfo('float').eps)

            probability_of_attribute_value = denominator / len(df)
            overall_entropy += -probability_of_attribute_value*entropy

        return abs(overall_entropy)
    
    else:
        # Create bins for the numerical values using the pandas.cut function
        bins = pd.cut(df[attribute], 3)
        # Find the unique bin values and the corresponding counts
        attribute_values = np.unique(bins)
        overall_entropy = 0

        for attribute_value in attribute_values:
            entropy = 0

            # Calculate entropy for each attribute value
            for target_class_value in target_class_values: 
                numerator = len(df[(attribute_value.left <= df[attribute]) & (df[attribute] <= attribute_value.right) & (df[target_class] == target_class_value)])
                denominator = len(df[(attribute_value.left <= df[attribute]) & (df[attribute] <= attribute_value.right)])
                probability = numerator / (denominator+np.finfo('float').eps)
                entropy += -probability * np.log2(probability + np.finfo('float').eps)
                
            probability_of_attribute_value = denominator / len(df)
            overall_entropy += -probability_of_attribute_value*entropy

        return abs(overall_entropy)


# Function calculates a weight for a given attribute in the primary dataframe based on 
# semantically similar attributes in a secondary dataframe    
def weight(attribute, df, secondary_df):
    # Create a list of semantic similarities between the input attribute and each attribute in the secondary dataframe
    secondary_keys = secondary_df.keys()[:-1]
    similarity_list = []
    
    for value in secondary_keys:
        similarity_list.append(phrase.semantic_similarity(attribute, value))

    most_similar = secondary_df.keys()[:-1][np.argmax(similarity_list)]
    
    # Calculate the information gain (IG) for the input attribute in the primary dataframe
    IG = entropy(df) - find_entropy_attribute(df, attribute)

    IG_secondary = entropy(secondary_df) - find_entropy_attribute(secondary_df, most_similar)

    # If the input attribute and the most similar attribute are semantically similar (similarity greater than 0.7), 
    # return a weight that is a function of the information gain in both dataframes
    if phrase.semantic_similarity(attribute, most_similar) > 0.7:
        return (IG + 0.5 * (IG_secondary - IG)) / IG
    else:
        return 1


# Finds the highest information gain attribute after considering adjustment from secondary dataframe
def find_winner(df, secondary_df):
    IG = []
    for key in df.keys()[:-1]:
        IG.append(weight(key, df, secondary_df) * (entropy(df) - find_entropy_attribute(df, key)))
    # Return the attribute with the highest information gain
    return df.keys()[:-1][np.argmax(IG)]


def get_subtable(df, node, value):
    if type(value) == str:
        return df[df[node] == value].reset_index(drop=True)
    else: 
        return df[(value.left <= df[node]) & (df[node] <= value.right)].reset_index(drop=True)


# Builds a decision tree by recursively splitting the df based on the attribute with the highest information gain
def build_tree(df, secondary_df, tree=None):
    best_attribute = find_winner(df, secondary_df)
    # Seperates string values from numeric
    if type(df[best_attribute][0]) == str:
        attribute_values = np.unique(df[best_attribute])
    else:
        # If numeric, bin the values into 3 equal parts
        bins = pd.cut(df[best_attribute], 3)
        attribute_values = np.unique(bins)

    if tree is None:
        tree={}
        tree[best_attribute] = {}

    for value in attribute_values:
        sub_table = get_subtable(df, best_attribute, value)
        # Get the unique values and count of the target attribute in the sub-dataframe
        class_values, counts = np.unique(sub_table[df.keys()[-1]], return_counts=True)
        # If the sub-dataframe is pure (only one value of the target attribute)
        if len(counts) == 1:
            tree[best_attribute][value] = class_values[0]
        else:
            # If the sub-dataframe is not pure, call the function recursively on the sub-dataframe
            tree[best_attribute][value] = build_tree(sub_table, secondary_df)
            
    return tree


# Takes a value and a leaf node of a tree, then outputs 
# the interval of the leaf node that the value is in
def intervalChooser(value, leaf, tree):
    intervals = tree[leaf].keys()

    for interval in intervals:
        # Check if the given value lies within the interval
        if interval.left <= value <= interval.right:
            return interval
        
    return None


# Function which traverses decision tree to make a decision
def predictor(test, tree, default=None):
    attribute = next(iter(tree)) 

    if type(test[attribute]) == str:

        if test[attribute] in tree[attribute].keys():
            result = tree[attribute][test[attribute]]

            if isinstance(result, dict): # Checks if result is of the dictionary class
                return predictor(test, result)
            else:
                return result
            
        else:
            return default
        
    else:
        interval = intervalChooser(test[attribute], attribute, tree)

        if interval is None:
            return default
        else:
            result = tree[attribute][interval]
            
            if isinstance(result, dict): # Checks if result is of the dictionary class
                return predictor(test, result)
            else:
                return result
            

# Translates the format of a CSV file to something readable for other functions
def row_to_dict(row):
    dict = {}
    for i, value in enumerate(row.index):
        dict[value] = row[i]
    return dict


# Function calculates the percentage of items in a list labeled correct
def calc_percentage(x):
    correct_count = 0
    for item in x:
        if item == 'correct':
            correct_count += 1
    percentage = (correct_count / len(x)) * 100
    return percentage


# Function calculates the accuracy of the model by comparing its
# predictions to actual values in a clean test dataframe
def accuracy(test_df, primary_df, secondary_df):
    results_list = []
    tree = build_tree(primary_df, secondary_df)

    for value in test_df.iloc:
        x = row_to_dict(value)
        x.pop(test_df.keys()[-1])

        if predictor(x, tree) == value[list(value.keys())[-1]]:
            results_list.append('correct')
        else:
            results_list.append('incorrect')
    
    return calc_percentage(results_list)
       

# Small data set from customer churn dataset #1
customer_churn1_small = pd.read_csv('customer_churn1_small.csv')
cc1_small_df = customer_churn1_small.drop(['Surname', 'Row Number', 'Customer Id', 'Geography'], axis=1) # Removed attributes which may not contribute to accurate prediction

# Large data set from customer churn dataset #1
customer_churn1_large = pd.read_csv('customer_churn1_large.csv')
cc1_large_df = customer_churn1_large.drop(['Surname', 'Row Number', 'Customer Id', 'Geography'], axis=1)

# Small data set from customer churn dataset #2
customer_churn2_small = pd.read_csv('customer_churn2_small.csv')
cc2_small_df = customer_churn2_small.drop(['Customer Id'], axis=1)

# Large data set from customer churn dataset #2
customer_churn2_large = pd.read_csv('customer_churn2_large.csv')
cc2_large_df = customer_churn2_large.drop(['Customer Id'], axis=1)

# Example of varibale format to predict output of a single data line:
# Data line only works with models trained on customer churn dataset #1
customer_churn1_test = {'Credit Score': 747, 'Gender': 'Female', 'Age': 41, 'Tenure': 5, 'Balance': 94521.17, 'Number Of Products': 2, 'Has Credit Card': 1, 'Is Active Member': 0, 'Estimated Salary': 194926.86}


###################################################################################################################################################################################################################


# Exmaple of how to print accuracy of various data set training combinations
# Keep in mine the order of parameters for the function accuracy(test_df, primary_df, secondary_df)
# NOTE: dealing with large datasets may take a few seconds
# NOTE: Parameters test_df and primary_df must belong to the same # dataset
print('The accuracy is:', accuracy(cc2_large_df, cc2_small_df, cc1_large_df)) # prints 69.72558604599489

# By setting parameters primary_df and secondary_df to the same value the program acts like a normal ID3 implementation
# without secondary dataset information mining (uncomment line below)
#print('The accuracy is:', accuracy(cc2_large_df, cc2_small_df, cc2_small_df)) # prints 63.3707365848239


###################################################################################################################################################################################################################


# Example of how to predict the value of a single variable:
# NOTE: dealing with large datasets may take a few seconds
tree = build_tree(cc1_small_df, cc2_large_df)
print('The predicted value is:', predictor(customer_churn1_test, tree)) # prints 0


###################################################################################################################################################################################################################
