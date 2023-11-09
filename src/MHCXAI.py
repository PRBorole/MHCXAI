#!/usr/bin/env python
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import shap
import sklearn
from sklearn.utils import check_random_state
import pickle
import argparse
import subprocess
import sys

class MHCXAI:
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
        train = np.genfromtxt(train_file, delimiter=',', dtype='<U20')
        train = np.array([self.AA_to_num(peptide) for peptide in train],dtype='<U20')
        return train.astype(float)

    def transform_peptide(self,peptide):
        peptide = [aa for aa in peptide]
        return self.AA_to_num(peptide).astype(float)

    def mhcflurry_predict_class(self, peptides_arr):
        # Some predictors need change of path therefore import only if needed. 
        from mhcflurry import Class1PresentationPredictor

        peptides_arr = [self.num_to_AA(instance) for instance in peptides_arr]
        predictor = Class1PresentationPredictor.load()
        df = predictor.predict(
            peptides=peptides_arr,
            alleles=[self.alleles],
            verbose=0)
        
        # MHCflurry allows presentation score and binding affinity as two outputs
        if self.mode=='affinity':
            label = 1 - np.log(df[self.mode])/np.log(50000)
        elif self.mode!=None:
            label = df[self.mode].to_numpy()
        else:
            print('ERROR: '+self.mode+' is not valid')

        if self.xai=='LIME':
            label_mat = np.zeros((len(label),2))
            label_mat[:,0] = 1-label
            label_mat[:,1] = label
            return label_mat

        elif self.xai=='SHAP':
            return label
        else:
            print('ERROR: '+self.xai+' is not valid')
            
    def netmhcpan_predict_class(self,peptides_arr):
        print(self.alleles[0])
        peptides_arr = [self.num_to_AA(instance) for instance in peptides_arr]
        if len(peptides_arr)==1:
            input_f = self.dest+self.peptide+"_temp_original_"+self.xai+".txt"
            f = open(input_f,"w")
            for p in peptides_arr:
                f.write(p+"\n")
            f.close()
        else:
            input_f = self.dest+self.peptide+"_mutations_original_"+self.xai+".txt"
            f = open(input_f,"w")
            for p in peptides_arr:
                f.write(p+"\n")
            f.close()    

        output_f = self.dest+self.peptide+"_NetMHCpan_out_"+self.xai+".xls"
        command = ["./netMHCpan-4.1/netMHCpan","-p", input_f,"-xls","-a", self.alleles,'-xlsfile', output_f,"-BA","-tdir",self.dest+"/netMHCpanXXXXXX"]
        result = subprocess.run(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output_pd = pd.read_csv(output_f,sep='\t',header = 1)
        
        label = output_pd[self.mode].to_numpy()
            
        label = np.array([1-(y*0.25) if y<=4 else 0 for y in label])

        if self.xai=='LIME':
            label_mat = np.zeros((len(label),2))
            label_mat[:,0] = 1-label
            label_mat[:,1] = label
            return label_mat

        elif self.xai=='SHAP':
            return label
        else:
            print('ERROR: '+self.xai+' is not valid')
        
    def mhcfovea_predict_class(self, peptides_arr):
        # Some predictors need change of path therefore import only if needed. 
        sys.path.append("./MHCfovea/")
        sys.path.append("./MHCfovea/mhcfovea")
        from mhcfovea import predictor_mhcxai
        
        peptides_arr = [self.num_to_AA(instance) for instance in peptides_arr]
        df = pd.DataFrame.from_dict({"sequence":peptides_arr,"mhc":len(peptides_arr)*[self.alleles]})
        output_df = predictor_mhcxai.main(input_file = df, output_dir = self.dest+"/tmp/")

        label = output_df[self.alleles].to_numpy()
        if self.xai=='LIME':
            label_mat = np.zeros((len(label),2))
            label_mat[:,0] = 1-label
            label_mat[:,1] = label
            return label_mat

        elif self.xai=='SHAP':
            return label
        else:
            print('ERROR: '+self.xai+' is not valid')
            
            
    def transphla_predict_class(self,peptides_arr):
        # Some predictors need change of path therefore import only if needed.
        sys.path.append('./transPHLA/TransPHLA-AOMP/TransPHLAAOMP/')
        import pHLAIformer
        
        hla_seq_pd = pd.read_csv('./transPHLA/TransPHLA-AOMP/Dataset/common_hla_sequence.csv')
        hla_seq = list(hla_seq_pd[hla_seq_pd['HLA']==allele]['HLA_sequence'])[0]
        
        peptides_arr = [self.num_to_AA(instance) for instance in peptides_arr]
        output_pd = pHLAIformer.transPHLA(peptide = peptides_arr, HLA = self.alleles, 
                  HLA_seq = hla_seq, output_dir = self.dest)
        
        label = output_pd['y_prob'].to_numpy()
        if self.xai=='LIME':
            label_mat = np.zeros((len(label),2))
            label_mat[:,0] = 1-label
            label_mat[:,1] = label
            return label_mat

        elif self.xai=='SHAP':
            return label
        else:
            print('ERROR: '+self.xai+' is not valid')
    
        
    def LIMEtabular(self,peptide,alleles,train_file,predictor,dest,mode=None,num_samples=25000):
        self.peptide = peptide
        self.peptide_size = len(peptide)
        self.class_names = ["0", "1"]
        self.feature_names = ['Pos'+str(i+1) for i in range(self.peptide_size)]
        self.categorical_features = range(self.peptide_size)
        self.categorical_names = {}
        self.alleles = alleles
        self.xai = 'LIME'
        self.mode = mode
        self.dest = dest
        
        values = list([x.split('=')[1] for x in self.AA])
        for i in self.categorical_features:
            self.categorical_names[i] = values
        
        
        train = self.transform_train(train_file)
        peptide = self.transform_peptide(peptide)

        explainer = lime.lime_tabular.LimeTabularExplainer(train, class_names=self.class_names, feature_names = self.feature_names,mode="classification",
                                                   categorical_features=self.categorical_features, 
                                                   categorical_names=self.categorical_names, kernel_width=3, verbose=False, random_state=42)
        
        if predictor=='mhcflurry':
            exp = explainer.explain_instance(peptide, self.mhcflurry_predict_class, num_features=len(peptide), num_samples=num_samples)
        elif predictor=='netmhcpan':
            exp = explainer.explain_instance(peptide, self.netmhcpan_predict_class, num_features=len(peptide), num_samples=num_samples)
        elif predictor=='mhcfovea':
            exp = explainer.explain_instance(peptide, self.mhcfovea_predict_class, num_features=len(peptide), num_samples=num_samples)
        elif predictor=='transphla':
            exp = explainer.explain_instance(peptide, self.transphla_predict_class, num_features=len(peptide), num_samples=num_samples)
        return explainer, exp
        
    def SHAPtabular(self,peptide,alleles,trainf_path,predictor,dest,mode=None,num_samples=500):
        self.peptide = peptide
        self.peptide_size = len(peptide)
        self.class_names = ["0", "1"]
        self.feature_names = ['Pos'+str(i) for i in range(self.peptide_size)]
        self.categorical_features = range(self.peptide_size)
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
        
        if predictor=='mhcflurry':
            explainer = shap.KernelExplainer(self.mhcflurry_predict_class, train_summary)
            shap_values = explainer.shap_values(peptide, nsamples=num_samples)
        elif predictor=='netmhcpan':
            explainer = shap.KernelExplainer(self.netmhcpan_predict_class, train_summary)
            shap_values = explainer.shap_values(peptide, nsamples=num_samples)
        elif predictor=='mhcfovea':
            explainer = shap.KernelExplainer(self.mhcfovea_predict_class, train_summary)
            shap_values = explainer.shap_values(peptide, nsamples=num_samples)
        elif predictor=='transphla':
            explainer = shap.KernelExplainer(self.transphla_predict_class, train_summary)
            shap_values = explainer.shap_values(peptide, nsamples=num_samples)
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

mhcxai = MHCXAI()

if xai=="LIME":
    print("LIME")
    explainer, exp = mhcxai.LIMEtabular(peptide,allele,trainf_path,predictor,dest,mode=mode,num_samples=25000)
    col_num = len(peptide) + 3
    lime_arr = np.zeros(col_num)
    lime_arr[0] = exp.intercept[1] # Intercept
    for i in range(0,len(peptide)):  # Weights
        idx = exp.as_list()[i][0][3]
        lime_arr[int(idx)] = exp.as_list()[i][1] 
    lime_arr[-2] = exp.score # R^2
    lime_arr[-1] = exp.local_pred # LIME model prediction
    np.save(dest+"/"+xai+"_"+peptide+"_"+allele+"_"+predictor+"_"+mode+".npy",lime_arr)

elif xai=="SHAP":
    print("SHAP")
    explainer, shap_values = mhcxai.SHAPtabular(peptide,allele,trainf_path,predictor,dest,mode=mode,num_samples=500)
    np.save(dest+"/"+xai+"_"+peptide+"_"+allele+"_"+predictor+"_"+mode+".npy",shap_values)

else:
    print("Incorrect XAI :",xai)