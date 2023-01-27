# Custom-ID3-Model
My take on the ID3 algorithm modified to increase prediction accuracy by using a similar secondary training dataset.

The model was designed to aid applications dealing with limited quantities of data, as such it takes in an ideally similar secondary training data
set then adjust information gain values used in the decision tree building process to increase accuracy.

All modifying theory and implementation is 100% custom

How to use:
1. Download all libraries used: nltk, numpy, pandas
2. Make sure all provided CSV files, 'Custom_ID3.py', and 'Phrase_WuPalmer.py' are all in the same location
2. Use the function 'predictor(test, tree)' with parameter 'tree' set to 'build_tree(df, secondary_df)' (Detailed in-code examples provided)
3. Use the function 'accuracy(test_df, primary_df, secondary_df)' with the parameter 'test_df' as a dataset you wish to test accuracy with (Detailed in-code examples provided)
