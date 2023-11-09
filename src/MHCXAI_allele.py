# Only for TransPHLA
import numpy as np
import pandas as pd
import subprocess
import sys
import lime
import lime.lime_tabular
import shap
import sklearn
from sklearn.utils import check_random_state
import argparse
import subprocess
import pickle

class MHCXAI_Allele:
    def __init__(self):
        self.AA = ['0=A','1=R','2=N','3=D','4=C','5=E','6=Q','7=G','8=H','9=I','10=L','11=K','12=M','13=F','14=P','15=S','16=T','17=W','18=Y','19=V','20=B','21=J','22=O','23=U','24=X','25=Z']
        
    def AA_to_num(self,peptide):
        values = dict([(x.split('=')[1], x.split('=')[0]) for x in self.AA])
        peptide = np.array(list(map(lambda x: values[x], peptide)))
        return peptide
        
    def num_to_AA(self,peptide):
        values = dict([(int(x.split('=')[0]), x.split('=')[1]) for x in self.AA])
        peptide = list(map(lambda x: values[x], peptide))
        str_peptide = ''
        for aa in peptide:
            str_peptide+=aa
        return str_peptide
    
    def transform_train(self,train_file):
        if type(train_file)==str:
            train = np.genfromtxt(train_file, delimiter=',', dtype='<U20')
        train = np.array([self.AA_to_num(peptide) for peptide in train_file],dtype='<U20')
        return train.astype(float)
    
    def transform_peptide(self,peptide):
        peptide = [aa for aa in peptide]
        return self.AA_to_num(peptide).astype(float)
    
    def transphla_predict_class(self,HLA_arr):
        # Some predictors need change of path therefore import only if needed.
        sys.path.append('/exports/csce/eddie/inf/groups/ajitha_project/piyush/transPHLA/TransPHLA-AOMP/TransPHLAAOMP/')
        import pHLAIformer
        
        HLA_arr = [self.num_to_AA(instance) for instance in HLA_arr]
        label = np.zeros(len(HLA_arr))
        for idx,hla in enumerate(HLA_arr):
            output_pd = pHLAIformer.transPHLA(peptide = [self.peptide], HLA = 'HLA', 
                  HLA_seq = hla, output_dir = self.dest)

            label[idx] = output_pd['y_prob'].item()
        
        if self.xai=='LIME':
            label_mat = np.zeros((len(label),2))
            label_mat[:,0] = 1-label
            label_mat[:,1] = label
            return label_mat

        elif self.xai=='SHAP':
            return label
        else:
            print('ERROR: '+self.xai+' is not valid')
    
    def LIMEtabular(self,peptide,alleles,alleles_seq,train_file,predictor,dest,mode=None,num_samples=25000):
        self.peptide = peptide
        self.alleles = alleles
        self.alleles_seq = alleles_seq
        self.allele_size = len(alleles_seq)
        self.class_names = ["0", "1"]
        self.feature_names = ['Pos'+str(i+1) for i in range(self.allele_size)]
        self.categorical_features = range(self.allele_size)
        self.categorical_names = {}
        self.xai = 'LIME'
        self.mode = mode
        self.dest = dest
        
        values = list([x.split('=')[1] for x in self.AA])
        for i in self.categorical_features:
            self.categorical_names[i] = values
            
        train_file = pd.read_csv(train_file).drop_duplicates('HLA_sequence')
        train = train_file['HLA_sequence'].to_list()
        train = self.transform_train(train)
        print(train.shape)
        alleles_seq = self.transform_peptide(alleles_seq)
        
        explainer = lime.lime_tabular.LimeTabularExplainer(train, class_names=self.class_names, feature_names = self.feature_names,
                                                   categorical_features=self.categorical_features,mode="classification", 
                                                   categorical_names=self.categorical_names, kernel_width=3, verbose=False,random_state=42)
        
        if predictor=='transphla':
            exp = explainer.explain_instance(alleles_seq, self.transphla_predict_class, num_features=len(alleles_seq), num_samples=num_samples)
        return explainer, exp
    
    def SHAPtabular(self,peptide,alleles,alleles_seq,train_file,predictor,dest,mode=None,num_samples=500):
        self.peptide = peptide
        self.alleles_seq = alleles_seq
        self.allele_size = len(alleles_seq)
        print(len(alleles_seq))
        self.class_names = ["0", "1"]
        self.feature_names = ['Pos'+str(i+1) for i in range(self.allele_size)]
        self.categorical_features = range(self.allele_size)
        self.categorical_names = {}
        self.alleles = alleles
        self.xai = 'SHAP'
        self.mode = mode
        self.dest = dest

        values = list([x.split('=')[1] for x in self.AA])
        for i in self.categorical_features:
            self.categorical_names[i] = values

        fileObj = open(trainf_path, 'rb')
        train_summary = pickle.load(fileObj)
        fileObj.close()
        peptide = self.transform_peptide(peptide)
        alleles_seq = self.transform_peptide(alleles_seq)
        if predictor=='transphla':
            explainer = shap.KernelExplainer(self.transphla_predict_class, train_summary)
            shap_values = explainer.shap_values(alleles_seq, nsamples=num_samples)
        return explainer, shap_values
    
    
parser = argparse.ArgumentParser(description="usage help",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_list", help="input peptide list file")
parser.add_argument("--trainf_path", help="Training File path")
parser.add_argument("--predictor", help="MHC predictor")
parser.add_argument("--xai", help="LIME/SHAP")
parser.add_argument("--mode", help="Name of the column: binding affinity, presentation score, Rank, etc.")
parser.add_argument("--dest", help="Destination location")
parser.add_argument("--index", help="Index of peptide in input list")

args = parser.parse_args()
config = vars(args)

input_list = args.input_list
trainf_path = args.trainf_path
mode = args.mode
predictor = args.predictor
xai = args.xai
index = int(args.index)
dest = args.dest

peptide_arr = pd.read_csv(input_list,index_col=False).peptide.to_list()
peptide = peptide_arr[index]

allele_arr = pd.read_csv(input_list,index_col=False).allele.to_list()
allele = allele_arr[index]

mhc_seq_pd = pd.read_csv('/exports/csce/eddie/inf/groups/ajitha_project/piyush/transPHLA/TransPHLA-AOMP/Dataset/common_hla_sequence.csv')
mhc_seq = list(mhc_seq_pd[mhc_seq_pd['HLA']==allele]['HLA_sequence'])[0]

mhcxai_allele = MHCXAI_Allele()

if xai=="LIME":
    print("LIME")
    explainer, exp = mhcxai_allele.LIMEtabular(peptide,allele,mhc_seq,trainf_path,predictor,dest,mode=mode,num_samples=30000)
    col_num = len(mhc_seq) + 3
    lime_arr = np.zeros(col_num)
    lime_arr[0] = exp.intercept[1] # Intercept
    for i in range(0,len(mhc_seq)):  # Weights
        pos = exp.as_list()[i][0].split('=')[0]
        if len(pos) == 5:
            idx = pos[-2:]
        elif len(pos) == 4:
            idx = pos[-1:]
        lime_arr[int(idx)] = exp.as_list()[i][1] 
    lime_arr[-2] = exp.score # R^2
    lime_arr[-1] = exp.local_pred # LIME model prediction
    np.save(dest+"/"+xai+"_"+peptide+"_"+allele+"_"+predictor+"_"+mode+"_allele.npy",lime_arr)

elif xai=="SHAP":
    print("SHAP")
    explainer, shap_values = mhcxai_allele.SHAPtabular(peptide,allele,mhc_seq,trainf_path,predictor,dest,mode=mode,num_samples=1000)
    np.save(dest+"/"+xai+"_"+peptide+"_"+allele+"_"+predictor+"_"+mode+"_allele.npy",shap_values)

else:
    print("Incorrect XAI :",xai)
