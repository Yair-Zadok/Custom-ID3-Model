# Custom-ID3-Model
My take on the ID3 algorithm, modified to increase prediction accuracy by using a similar secondary training dataset.

The model was designed to aid applications dealing with limited quantities of data, as such it takes in a large similar secondary training data
set, then adjust information gain values used in the decision tree building process to increase accuracy.

All modifying theory and implementation is 100% original.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Modifications to ID3:
The information gain values of attributes (topic categories) can be unreliable with little data, as such the program searches for semantically similar attributes in a larger similar secondary dataset, then adjusts the information gain values of the original attributes to be closer to the secondary attribute. By doing this the sorting of the decision is influenced by a much larger data set hence proving more accurate. 

Why not simply use the secondary dataset as the primary set? 
Because attributes (topic categories) are most likely different in the primary and secondary datasets, meaning that a small company trying to predict values will not have the same attributes needed to use 100% of a larger secondary dataset. 

Accuracy Readings:
No rigourous testing was done, however by experimentation I noticed a roughly 5-7% increase in accuracy with the model using a secondary dataset compared to not.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

How to use:
1. Download all libraries used: nltk, numpy, pandas
2. Make sure all provided CSV files, 'Custom_ID3.py', and 'Phrase_WuPalmer.py' are all in the same location
2. Use the function 'predictor(test, tree)' with parameter 'tree' set to 'build_tree(df, secondary_df)' (Detailed in-code examples provided)
3. Use the function 'accuracy(test_df, primary_df, secondary_df)' with the parameter 'test_df' as a dataset you wish to test accuracy against (Detailed in-code examples provided)
