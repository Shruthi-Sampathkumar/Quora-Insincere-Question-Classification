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

How to run unbiased embeddings generator?

Here we are trying to remove the bias from the Google News embeddings

The following are the scripts being used:
learn_gender_specific.py: It has a list of gender-specific words
debias.py: We give the biased embedding, gender-pairs, gender-specific words, and pairs to equalize and we get an unbiased version of the embedding

The words:
gender_specific_seed.json: gender-specific words list
gender_specific_full.json: gender-specific words list
definitional_pairs.json: The word pairs used for computing the gender direction
equalize_pairs.json: The word pairs that represent gender direction

The following are the steps:
Step1: Download the folder 'Dbias_embeddings'
Step2: Go into dbiaswe folder in the Dbias_embeddings folder
Step3: Download the google embeddings 
Step4: Run the command --> python debias.py ../embeddings/GoogleNews-vectors-negative300.bin ../data/definitional_pairs.json ../data/gender_specific_full.json ../data/equalize_pairs.json ../embeddings/GoogleNews-vectors-negative300-hard-debiased.bin
