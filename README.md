# MHCXAI
Framework for generating local SHAP and LIME explanations for MHC class I predictors

<p align="center"><img src="figures/MHCXAI.png" alt="" width="800"></p>

MHC class I predictors supported:
1. MHCflurry (https://doi.org/10.1016/j.cels.2020.06.010)
2. NetMHCpan (https://doi.org/10.1093/nar/gkaa379)
3. MHCfovea (https://doi.org/10.1038/s42003-021-02716-8)
4. TransPHLA (https://doi.org/10.1038/s42256-022-00459-7)

# Notebooks

**0 - Benchmark**: Contains benchmark results for MHC-Bench and MHC-Bench-2

**1 - MHCXAI usage and instance based explanations**: Demonstrates usage of MHCXAI for the above listed predictors

**2 - Model Validation with BAlaS**: Figures used in paper for testing *validity*

**3 - Consistency**: Figures used in paper for *consistency*

**4 - Stability**: Figures used in paper for *stability*

**5 - Explanation for alleles - TransPHLA**: Usage of MHCXAI adapted to generate explanations for allele sequence

# Folders
**1 - example**: Contains files used by notebooks for creating figures

**2 - data**: Contains results for benchmark, model validation, stability and consistency along with MHC-Bench dataset
1. The *MHC-Bench* folder contains the benchmark dataset divided per allele
2. The *MHC-Bench-v2* folder contains modified MHC-Bench such peptides overlapping with training data is removed.
While using, combine all the allele files into one name them MHC-Bench.csv or MHC-Bench-vs.csv.

# To add new predictor
1. In MHCXAI.py, create a function \<predictor\>_predict_class(self,peptides_arr) which accepts peptides list under the class MHCXAI.
2. Add the path and import the predictor under this new function
3. For LIME, the return should be a matrix of two columns containing probability for non-binder or negative class in first column and probability for binder or positive class in second column 
4. For SHAP, the return should be an array of probability for binder or positive class
5. Add the function name in LIMEtabular and SHAPtabular
6. Make sure to process the training data for the predictor


Citation:
```bibtex

@article{borole2024building,
  title={Building trust in deep learning-based immune response predictors with interpretable explanations},
  author={Borole, Piyush and Rajan, Ajitha},
  journal={Communications biology},
  volume={7},
  number={1},
  pages={279},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

