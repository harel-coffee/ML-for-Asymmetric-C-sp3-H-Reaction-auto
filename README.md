# ML-for-Asymmetric-C-sp3-H-Reaction
Machine Learning Approaches for Catalytic Asymmetric beta-C(sp3)-H Bond Activation Reactions
ML Models such as Gaussian process regression (GPR), k-nearest neighbour (k-NN), random forest (RF), gradient boosting (GB), and decision tree (DT) were created using the scikit-learn Python machine learning package. The deep neural network (DNN) model is created using PyTorch, a deep learning framework. The following initialization requirements must be met before performing actual calculations: (a) Python3 installation on a suitable configuration server, (b) creation of necessary folders, and (c) placement of the data file in the appropriate folder. Steps involved in running the code:

   Step 1: Generation of Dataset 

In ‘data’ folder, you can see three different subfolder corresponding to MLS model, unbound model, and out-of-bag (OOB) model. Each subfolder contains an xlsx-formatted excel file dataset. Every detailed information about the reaction component, reaction condition and collected features as well as target values (%ee) are provided in excel file. In the supporting information, the identities of each reaction entity are listed. Their optimized geometry is used to collect features (detailed discussion in supporting information). Additionally, each dataset contained synthetically samples which is generated synthetic minority oversampling technique (SMOTE: see Generation of synthetic samples). The datasets corresponding to the MLS model and unbound model comprise individual sets (LA, LB, LC, and LD) as well as extra combination sets (LA-LB, LA-LB-LC, and LA-LB-LC-LD). 

• Take just the features and label ('%ee') from the 'xlsx' file and save it as a .csv file, as csv-formatted file is needed to execute the code. Furthermore, it has already provided 'csv' files of both real ('MLS-LA-LB-LC-LD-Real.csv') and real plus synthetic ('MLS-LA-LB-LC-LD-Real-Synthetic.csv') data for individual sets and combined sets. In similar way, you can generate csv file for out-of-bag set, and all three sets ('set-1.csv', 'set-1.csv', 'set-1.csv') are provided.

    Generation of Synthetic Samples:
          
To include synthetic data, we used the SMOTE technique. It has done by using SMOTE.ipynb python file (code/SMOTE/SMOTE.ipynb).  To run this piece of code, you need real dataset (e.g. 'MLS-LA-LB-LC-LD-Real.csv') and you have to mention %ee value up to which you want to add synthetic samples (These are commented in .ipynb file). It will create the csv file containing both real and synthetic data. In this study, we have generated synthetic samples in the minority class in the 0-80 class boundary using the SMOTE (borderline-2).

   Step 2: Python Script to Run Code

Place the data files (e.g. 'MLS-LA-LB-LC-LD-Real.csv' and 'MLS-LA-LB-LC-LD-Real-Synthetic.csv') and python file (e.g. GPR_RBF_synthetic.py) in the working directory to run the code. Change the necessary file names and save .py file before running the code. Type "python GPR_RBF_synthetic.py > MLS-LA-LB-LC-LD-Real-Synthetic-Result.txt" in the working directory. The 'MLS-LA-LB-LC-LD-Real-Synthetic-Result.txt' file contains the final test and train RMSE, which is averaged over 100 different randomized test–train splits.

Different ML algorithms, such as k-NN, RF, GB, DNN, and DT, require a similar procedure. There is a .py file for each of them.

Out-of-bag Testing

Place the python file (e.g. Out-of-bag.py), csv file containing both real plus synthetic samples (e.g. 'MLS-LA-LB-LC-LD-Real-Synthetic.csv'), csv file of any OOB set (e.g. 'set-1.csv') in working directory. Change the filenames in python file, change the optimal hyper-parameter value, after modification save .py file and then run the python script "python Out-of-bag.py > set-1-Result.txt". 'set-1-Result.txt' will store test RMSE for OOB set.
