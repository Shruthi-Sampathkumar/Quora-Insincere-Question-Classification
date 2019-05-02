# Quora-Insincere-Question-Classification
Link to competition: https://www.kaggle.com/c/quora-insincere-questions-classification

The final model is final_model.kernel.ipynb

# How to run the final model?
Follow below steps to run the best model with F-1 score 0.6635

Step 1: Open a Kaggle Kernel

Step 2: Upload Quora-Insincere-Question-Classification dataset to your kernel

Step 3: Upload the IDF.mat file on the Kaggle kernel using the "Upload Dataset" option and give the Dataset title as "idf-mat-file". (Note: Please keep the title same as paths have been set accordingly in the code) 

Step 4: Upload final_model_kernel.ipynb

Step 5: Turn GPU, Internet on for that kernel

Step 6: Run all

Step 7: F-1 score is printed in the end.

------------------------------------------------------------------------------------------------------------------------------------------
# How to run idf generator?

Step 1: Upload the idf_generator.py into a Kaggle kernel.
Step 2: Upload Quora-Insincere-Question-Classification dataset to your kernel
Step 3: Execute code
Step 4: Extract IDF.mat from output folder of your kernel.



----------------------------------------------------------------------------------------------------------------------------------------

# How to run different models?


Step 1: Download data from https://www.kaggle.com/c/quora-insincere-questions-classification/data

Step 2: Unzip it into data folder in repository

Step 4: Download embeddings matrix directly from https://drive.google.com/open?id=1OwL0TSafpvvf0BN_ltz3o7LHrpZJZm97 
        <to reduce computation time>
Note: this drive link is only accessible to TAMU ids.
        embedding.mat => 4 layer embeddings <glove, google news, wiki news, paragram>
        embedding2.mat => 4 layer embeddings <glove, google news, wiki news, paragram> + idf

Step 5: Place the 2 extracted .mat files into the data folder.

Step 6: Pick up any of the models notebook/python file to train and evaluate on this data.
        
        To run any model, execute: "python <model_name>.py" for example "python isr_idf_no_cnn_no_attn_mat.py"
----------------------------------------------------------------------------------------------------------------------------------------
# How to run unbiased embeddings generator?
