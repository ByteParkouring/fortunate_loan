Water quality:

kaggle dataset was bad (optimized accuracy of 67% while null accuracy also above 60%)

--> different dataset: https://zenodo.org/records/7056647 GRQA_data_v1.3.zip

take only files with site_country "Germany" and respect only german values.

Respect only files for which quality standard ranges can be found online (the .csv files themselves don't make statement about potability)

In most cases newer than 2000 is valid, but for COD and BOD 1990 is necessary. COD and BOD also have no rhein

Since unique site_names seem too similar and there are hundreds and thousands of them, we won't filter for one specific site name, but just respect all and hope that the data is balanced across the files

DOES PREDICTION OF POTABILITY EVEN MAKE SENSE? JUST DEFINE THRESHOLD OF ALLOWED INTERVALS AND ITERATE IF ALL TRUE



LOAN CLASSIFICATION OVER 90% ACCURACY:
- notebook: https://www.kaggle.com/code/abdallahabuelftouh/loan-approval-classification-logistic-svc-and-knn
- dataset: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data