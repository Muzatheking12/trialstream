import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdchem, QED, rdmolops, rdMolDescriptors
import glob
import pandas as pd
from padelpy import padeldescriptor
from joblib import load
from rdkit.Chem import AllChem, MACCSkeys,  AtomPairs, EState, Pharm2D
from rdkit.Chem.rdmolops import PatternFingerprint
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Avalon import pyAvalonTools
from compchemkit import fingerprints
import numpy as np
import xgboost
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, mean_squared_error, matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from streamlit_ketcher import st_ketcher
from shap import TreeExplainer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from rdkit.Chem import DataStructs
from rdkit.DataStructs import ExplicitBitVect




import os

filepath = os.path.dirname(__file__)
xml = os.path.join(filepath, r'xml/*.xml')
title = os.path.join(filepath, 'title.txt')
image = os.path.join(filepath, 'hello.png')
about = os.path.join(filepath, 'About.txt')
model = os.path.join(filepath, 'model.txt')
data = os.path.join(filepath, 'data.txt')
dataimg  = os.path.join(filepath, 'plot.png')
citation = os.path.join(filepath, 'citation.txt')
citationlink = os.path.join(filepath, 'citationlink.txt')
author = os.path.join(filepath, 'authors.txt')
smiledraw = os.path.join(filepath, 'mol.png')
rmodel = os.path.join(filepath, 'model.joblib')
smilefile = os.path.join(filepath, 'molecule.smi')
fingerprint_output_file = os.path.join(filepath, "fingerprint.csv")
fingerprint_output_file_txt = os.path.join(filepath, "fingerprint.csv.log")
cfp = os.path.join(filepath, 'cfp.txt')
loadm = os.path.join(filepath, 'model.joblib')
xcol = os.path.join(filepath, 'col.csv')
std = os.path.join(filepath, 'std.txt')
task = os.path.join(filepath, 'task.txt')
above = os.path.join(filepath, 'above.txt')
below = os.path.join(filepath, 'below.txt')
smarts = os.path.join(filepath, 'SMARTS_LIST_NEW.csv')
TPNG = os.path.join(filepath, 'tanimoto.png')
TANPER = os.path.join(filepath , 'TANPER.txt')
ADBoundpath = os.path.join(filepath, 'ADBoundBox.xlsx')



st.sidebar.title('*Input SMILES*')

with open(title, 'r') as file:
    content = file.read()


st.title(f'  *{content}* :pill:   ')

tab0 ,tab1, tab2, tab3, tab4, tab5 = st.tabs(['Predict' , 'About', 'Dataset', 'Model', 'Citation', 'Authors'])


with tab0:
    
    st.write("""### Instructions""")
    st.write('- *Input **SMILES** in the Sidebar* ')
    with open(std, 'r') as file:
        content = file.read()
    st.write(f'- *Press Confirm Button To Generate Predicted **{content}** value*')
    st.write(" - *Draw Your Structure and Click on Apply to Generate SMILES*")
    SMI = st_ketcher()
    st.write("""## Output""")
    SMILES_input = st.sidebar.text_input(' **Enter Your SMILES Below** ', SMI)
    button = st.sidebar.button('Confirm')
    st.sidebar.title('*Input Excel Sheet*')
    uploaded_file = st.sidebar.file_uploader("Upload Excel Sheet For Multiple Molecule Processing\n Headers -> ['SMILES']", type=["xlsx"])
    button2 = st.sidebar.button('Predict')
    with open(TANPER, 'r') as file:
        content = file.read()
        content = float(content)
    AD_slider = st.sidebar.slider('Set Tanimoto Similarity Threshold for Applicability Domain', min_value=0.0, max_value=1.0, value=content, step=0.01)

    # def tanimoto(fp1, fp2):
    #     intersection = np.sum(np.bitwise_and(fp1, fp2))
    #     union = np.sum(np.bitwise_or(fp1, fp2))

    #     if union == 0:
    #         return 0
    #     else:
    #         return intersection / union
    def tanimoto(fp1, fp2):

        # convert numpy row -> RDKit bitvector
        def to_bitvect(arr):
            arr = np.asarray(arr).ravel().astype(int)
            bv = ExplicitBitVect(len(arr))
            for i, bit in enumerate(arr):
                if bit:
                    bv.SetBit(i)
            return bv

        bv1 = to_bitvect(fp1)
        bv2 = to_bitvect(fp2)

        return DataStructs.BulkTanimotoSimilarity(bv1, [bv2])[0]

    def compute_fingerprints(smiles_list, fp_type):

        fp_list = []
        mol_list = []
        bitinfo_list = []

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            if fp_type == 'Morgan':
                radius = 2
                nBits = 1024
                bitInfo = {}

                fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius,
                    nBits=nBits,
                    bitInfo=bitInfo
                )

                mol_list.append(mol)
                bitinfo_list.append(bitInfo)

            elif fp_type == 'TopologicalTorsion':
                fingerprint = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)

            elif fp_type == 'AvalonFP':
                fingerprint = pyAvalonTools.GetAvalonFP(mol)

            elif fp_type == 'PatternFP':
                fingerprint = PatternFingerprint(mol)

            fp_list.append(list(fingerprint.ToBitString()))

        # Save to CSV as before
        fp_df = pd.DataFrame(fp_list, columns=[f"FP_{i+1}" for i in range(len(fp_list[0]))])
        fp_df.to_csv(fingerprint_output_file, index=False)

      
        if fp_type == 'Morgan':
            return mol_list, bitinfo_list
    def model_predexcel():
         if uploaded_file:
             
              dfx = pd.read_excel(uploaded_file)
              if 'SMILES' in dfx.columns:
               st.write("*Predicted Data Table*")
               if 'Value' in dfx.columns or 'Bioactivity' in dfx.columns:
                Smiles = dfx['SMILES']
                predictions_list = []
                for i, smi in enumerate(Smiles):
                     m = Chem.MolFromSmiles(smi, sanitize=False)
                     if m is None:
                          st.write(" ### Your **SMILES** is not correct  ")
                     else:
                        with open(task, 'r') as file:
                            con = file.read()
                            file.close()
                        if con == 'Regression':
                            xml_files = glob.glob(xml)
                            xml_files.sort() 
                            FP_list = ['AtomPairs2DCount',
                                'AtomPairs2D',
                                'EState',
                                'CDKextended',
                                'CDK',
                                'CDKgraphonly',
                                'KlekotaRothCount',
                                'KlekotaRoth',
                                'MACCS',
                                'PubChem',
                                'SubstructureCount',
                                'Substructure']
                            
                        
                            fp = dict(zip(FP_list, xml_files))
                            df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                            df.to_csv(smilefile, sep='\t', index=False, header=False)
                            with open(cfp, 'r') as file:
                                content = file.readline()
                            fingerprint = content
                            if fingerprint in FP_list:
                                fingerprint_descriptortypes = fp[fingerprint]
                                padeldescriptor(mol_dir= smilefile, 
                                            d_file=fingerprint_output_file, #'Substructure.csv'
                                            #descriptortypes='SubstructureFingerprint.xml', 
                                            descriptortypes= fingerprint_descriptortypes,
                                            detectaromaticity=True,
                                            standardizenitro=True,
                                            standardizetautomers=True,
                                            threads=2,
                                            removesalt=True,
                                            log=True,
                                            fingerprints=True)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Value'], axis=1)
                                fingerprintx = R.values
                                X = descriptors.drop(['Name'], axis=1)
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                

                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'
                                    
                        

                               
                                
                        
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                predx = round(float(pred[0]), 2)
                               
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                
                                if 'Value' in dfx.columns:

                                    true_value = dfx.loc[i, 'Value']
                    
                                    predictions_list.append({'SMILES': smi, 'Predicted Value': predx, 'True Value': true_value, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                   

                                   
                                else:
                                    
                                    st.write(f" ### Give Header As Value . This is Regression Model")
                                
                                    

                            elif fingerprint == 'CSFP':
                        
                                df = pd.DataFrame({'SMILES': [smi], 'Name': [i]})
                                df.to_csv(smilefile, sep='\t', index=False, header=False)

                                smiles = df['SMILES']

                                # Load SMARTS patterns
                                SMARTS = np.loadtxt(smarts, dtype=str, comments=None)
                                CSFP = fingerprints.FragmentFingerprint(substructure_list=SMARTS.tolist())

                                # Transform SMILES to fingerprints
                                data_csfp = CSFP.transform_smiles(smiles)

                                # Convert sparse matrix to dense array
                                data_csfp_dense = data_csfp.toarray()

                                # Convert dense array to DataFrame
                                data_csfp_df = pd.DataFrame(data_csfp_dense)
                                
                                del data_csfp_df[data_csfp_df.columns[0]]

                                # Ensure there is no 'Unnamed: 0' column by setting index=False when saving
                                data_csfp_df.to_csv(fingerprint_output_file, index=False)

                                # Read descriptors from the CSV file
                                descriptors = pd.read_csv(fingerprint_output_file)

                                # Ensure no 'Unnamed: 0' column is expected in subsequent operations
                                

                                # Read X1 and drop 'Value' column
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Value'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]  
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'   
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                predx = round(float(pred[0]), 2)
                               
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                
                                if 'Value' in dfx.columns:
                                    true_value = dfx.loc[i, 'Value']
                    
                                    predictions_list.append({'SMILES': smi, 'Predicted Value': predx, 'True Value': true_value, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                   

                                   
                                else:
                                    
                                    st.write(f" ### Give Header As Value . This is Regression Model")
                              
                               
                            elif fingerprint == 'RDKitDescriptors':
                                descriptor_names = [desc[0] for desc in Descriptors.descList]
                                calculator =MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                def compute_all_descriptors(smiles):
                                    descriptor_names = [desc[0] for desc in Descriptors.descList]
                                    calculator =MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        # Compute descriptors
                                        return pd.Series(calculator.CalcDescriptors(mol))
                                    else:
                                        # Return NaN for invalid SMILES
                                        return pd.Series([None] * len(descriptor_names))
                                df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                descriptors_df = df['SMILES'].apply(compute_all_descriptors)
                                descriptors_df = pd.DataFrame(descriptors_df.values.tolist(), columns=descriptor_names)
                                
                                nan_columns = descriptors_df.columns[descriptors_df.isna().any()].tolist()

                               

                                # Drop columns with NaN values
                                descriptors_df.drop(columns=nan_columns, inplace=True)
                                float32_max = np.finfo(np.float32).max
                                float32_min = np.finfo(np.float32).min
                                descriptors_df.replace([np.inf, -np.inf], [float32_max, float32_min], inplace=True)

                                # Clip values to fit float32 range
                                descriptors_df = descriptors_df.clip(lower=float32_min, upper=float32_max)

                                # Convert to float32
                                descriptors_df = descriptors_df.astype('float32')
                                descriptors_df.to_csv(fingerprint_output_file,  index=False)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Value'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                dataxx = pd.read_excel(ADBoundpath, index_col=0)
                                dataxx = dataxx[R.columns]
                                min_bounds = dataxx.loc['Min']
                                max_bounds = dataxx.loc['Max']
                                compound_descriptors = X.iloc[0].to_dict()  # Assuming single compound; adjust if multiple
                                descriptor_results = {}
                                AD = "IN Applicability Domain"  # Default to "IN"
                                tolerance = 1e-6
                                for descriptor, value in compound_descriptors.items():
                                    if descriptor in min_bounds.index:
                                        is_within_range = min_bounds[descriptor] <= value <= max_bounds[descriptor] 
                                        descriptor_results[descriptor] = "Within Range" if is_within_range else "Out of Range"

                                        # Update overall status if any descriptor is out of range
                                        if not is_within_range:
                                            AD = "OUT Applicability Domain"
                                    else:
                                        descriptor_results[descriptor] = "Descriptor not found in bounds data"  
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                predx = round(float(pred[0]), 2)
                              
                                with open(std, 'r') as file:
                                    content = file.read()
                                
                             
                               
                                if 'Value' in dfx.columns:
                                    true_value = dfx.loc[i, 'Value']
                    
                                    predictions_list.append({'SMILES': smi, 'Predicted Value': predx, 'True Value': true_value, 'AD': AD})
                                   

                                   
                                else:
                                    
                                    st.write(f" ### Give Header As Value . This is Regression Model")
                            else:
                                df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                smiles = df['SMILES'] 
                                compute_fingerprints(smiles, fingerprint)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Value'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'
                                
                              
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                predx = round(float(pred[0]), 2)
                              
                                with open(std, 'r') as file:
                                    content = file.read()
                                
                             
                               
                                if 'Value' in dfx.columns:
                                    true_value = dfx.loc[i, 'Value']
                    
                                    predictions_list.append({'SMILES': smi, 'Predicted Value': predx, 'True Value': true_value, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                   

                                   
                                else:
                                    
                                    st.write(f" ### Give Header As Value . This is Regression Model")
                               
                        

                        else:
                            with open(cfp, 'r') as file:
                                    content = file.readline()
                            fingerprint = content
                            FP_list = ['AtomPairs2DCount',
                                    'AtomPairs2D',
                                    'EState',
                                    'CDKextended',
                                    'CDK',
                                    'CDKgraphonly',
                                    'KlekotaRothCount',
                                    'KlekotaRoth',
                                    'MACCS',
                                    'PubChem',
                                    'SubstructureCount',
                                    'Substructure']
                                
                            if fingerprint in FP_list:
                                xml_files = glob.glob(xml)
                                xml_files.sort() 
                                FP_list = ['AtomPairs2DCount',
                                    'AtomPairs2D',
                                    'EState',
                                    'CDKextended',
                                    'CDK',
                                    'CDKgraphonly',
                                    'KlekotaRothCount',
                                    'KlekotaRoth',
                                    'MACCS',
                                    'PubChem',
                                    'SubstructureCount',
                                    'Substructure']
                                
                            
                                fp = dict(zip(FP_list, xml_files))
                                df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                fingerprint_descriptortypes = fp[fingerprint]
                                padeldescriptor(mol_dir= smilefile, 
                                            d_file=fingerprint_output_file, #'Substructure.csv'
                                            #descriptortypes='SubstructureFingerprint.xml', 
                                            descriptortypes= fingerprint_descriptortypes,
                                            detectaromaticity=True,
                                            standardizenitro=True,
                                            standardizetautomers=True,
                                            threads=2,
                                            removesalt=True,
                                            log=True,
                                            fingerprints=True)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Bioactivity'], axis=1)
                                fingerprintx = R.values
                                X = descriptors.drop(['Name'], axis=1)
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'
                    
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                value = str(pred[0])
                                if value == '1':
                                    with open(below, 'r') as file:
                                        man = file.read()
                                  
                                  
                                    if 'Bioactivity' in dfx.columns:
                                      true_value = dfx.loc[i, 'Bioactivity']
                    
                                     
                                      predictions_list.append({'SMILES': smi, 'Predicted Activity': man, 'True Activity': true_value, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                   

                                   
                                    else:
                                    
                                     st.write(f" ### Give Header As Bioactivity . This is Classification Model")
                                   
                            

                                else:
                                    with open(above, 'r') as file:
                                        man = file.read()
                                    if 'Bioactivity' in dfx.columns:
                                      true_value = dfx.loc[i, 'Bioactivity']
                    
                                      predictions_list.append({'SMILES': smi, 'Predicted Activity': man, 'True Activity': true_value, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})

                                   
                                    else:
                                    
                                     st.write(f" ### Give Header As Bioactivity . This is Classification Model")
                                 
                                   
                                  
                            elif fingerprint == 'CSFP':
                        
                                df = pd.DataFrame({'SMILES': [smi], 'Name': [i]})
                                df.to_csv(smilefile, sep='\t', index=False, header=False)

                                smiles = df['SMILES']

                                # Load SMARTS patterns
                                SMARTS = np.loadtxt(smarts, dtype=str, comments=None)
                                CSFP = fingerprints.FragmentFingerprint(substructure_list=SMARTS.tolist())

                                # Transform SMILES to fingerprints
                                data_csfp = CSFP.transform_smiles(smiles)

                                # Convert sparse matrix to dense array
                                data_csfp_dense = data_csfp.toarray()

                                # Convert dense array to DataFrame
                                data_csfp_df = pd.DataFrame(data_csfp_dense)
                                
                                del data_csfp_df[data_csfp_df.columns[0]]

                                # Ensure there is no 'Unnamed: 0' column by setting index=False when saving
                                data_csfp_df.to_csv(fingerprint_output_file, index=False)

                                # Read descriptors from the CSV file
                                descriptors = pd.read_csv(fingerprint_output_file)

                                # Ensure no 'Unnamed: 0' column is expected in subsequent operations
                                

                                # Read X1 and drop 'Value' column
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Bioactivity'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'
                                
                               
                                model = load(loadm)
                                
                                pred = model.predict(X)
                           
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                value = str(pred[0])
                                if value == '1':
                                    with open(below, 'r') as file:
                                        man = file.read()
                                    
                                    if 'Bioactivity' in dfx.columns:
                                      true_value = dfx.loc[i, 'Bioactivity']
                    
                                     
                                      predictions_list.append({'SMILES': smi, 'Predicted Activity': man, 'True Activity': true_value, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                   
                                    else:
                                    
                                     st.write(f" ### Give Header As Bioactivity . This is Classification Model")
                                   
                                
                                  
                                   
                                else:
                                    with open(above, 'r') as file:
                                        man = file.read()
                                  


                                    if 'Bioactivity' in dfx.columns:
                                      true_value = dfx.loc[i, 'Bioactivity']
                    
                                   
                                      predictions_list.append({'SMILES': smi, 'Predicted Activity': man, 'True Activity': true_value, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                   
                                    else:
                                    
                                     st.write(f" ### Give Header As Bioactivity . This is Classification Model")

                            elif fingerprint == 'RDKitDescriptors':

                                descriptor_names = [desc[0] for desc in Descriptors.descList]
                                calculator =MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                def compute_all_descriptors(smiles):
                                    descriptor_names = [desc[0] for desc in Descriptors.descList]
                                    calculator =MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        # Compute descriptors
                                        return pd.Series(calculator.CalcDescriptors(mol))
                                    else:
                                        # Return NaN for invalid SMILES
                                        return pd.Series([None] * len(descriptor_names))
                                df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                descriptors_df = df['SMILES'].apply(compute_all_descriptors)
                                descriptors_df = pd.DataFrame(descriptors_df.values.tolist(), columns=descriptor_names)
                                
                                nan_columns = descriptors_df.columns[descriptors_df.isna().any()].tolist()

                               

                                # Drop columns with NaN values
                                descriptors_df.drop(columns=nan_columns, inplace=True)
                                float32_max = np.finfo(np.float32).max
                                float32_min = np.finfo(np.float32).min
                                descriptors_df.replace([np.inf, -np.inf], [float32_max, float32_min], inplace=True)

                                # Clip values to fit float32 range
                                descriptors_df = descriptors_df.clip(lower=float32_min, upper=float32_max)

                                # Convert to float32
                                descriptors_df = descriptors_df.astype('float32')
                               
                                descriptors_df.to_csv(fingerprint_output_file,  index=False)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Bioactivity'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                dataxx = pd.read_excel(ADBoundpath, index_col=0)
                                dataxx = dataxx[R.columns]
                                min_bounds = dataxx.loc['Min']
                                max_bounds = dataxx.loc['Max']
                                compound_descriptors = X.iloc[0].to_dict()  # Assuming single compound; adjust if multiple
                                descriptor_results = {}
                                AD = "IN Applicability Domain"  # Default to "IN"
                                 # Default to "IN"
                                tolerance = 1e-6
                                for descriptor, value in compound_descriptors.items():
                                    if descriptor in min_bounds.index:
                                        is_within_range = min_bounds[descriptor] <= value <= max_bounds[descriptor]
                                        descriptor_results[descriptor] = "Within Range" if is_within_range else "Out of Range"

                                        # Update overall status if any descriptor is out of range
                                        if not is_within_range:
                                            AD = "OUT Applicability Domain"
                                    else:
                                        descriptor_results[descriptor] = "Descriptor not found in bounds data"     
                                model  = load(loadm)
                                
                                pred = model.predict(X)
                           
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                value = str(pred[0])
                                if value == '1':
                                    with open(below, 'r') as file:
                                        man = file.read()
                                    
                                    if 'Bioactivity' in dfx.columns:
                                      true_value = dfx.loc[i, 'Bioactivity']
                    
                                     
                                      predictions_list.append({'SMILES': smi, 'Predicted Activity': man, 'True Activity': true_value, 'AD': AD})
                                   
                                    else:
                                    
                                     st.write(f" ### Give Header As Bioactivity . This is Classification Model")
                                   
                                
                                  
                                   
                                else:
                                    with open(above, 'r') as file:
                                        man = file.read()
                                  


                                    if 'Bioactivity' in dfx.columns:
                                      true_value = dfx.loc[i, 'Bioactivity']
                    
                                   
                                      predictions_list.append({'SMILES': smi, 'Predicted Activity': man, 'True Activity': true_value, 'AD': AD})
                                   
                                    else:
                                    
                                     st.write(f" ### Give Header As Bioactivity . This is Classification Model")
 
                                  
                                 
                            else:
                                df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                smiles = df['SMILES']
                                compute_fingerprints(smiles, fingerprint)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Bioactivity'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'
                                
                               
                                model  = load(loadm)
                                
                                pred = model.predict(X)
                              
                            
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                value = str(pred[0])
                                if value == '1':
                                    with open(below, 'r') as file:
                                        man = file.read()

                                    if 'Bioactivity' in dfx.columns:
                                      true_value = dfx.loc[i, 'Bioactivity']
                                 
                                      predictions_list.append({'SMILES': smi, 'Predicted Activity': man, 'True Activity': true_value, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})

                                   
                                    else:
                                    
                                     st.write(f" ### Give Header As Bioactivity . This is Classification Model")
                                    
                                  
                                  
                                 
                                
                                else:
                                    with open(above, 'r') as file:
                                        man = file.read()

                                    if 'Bioactivity' in dfx.columns:
                                      true_value = dfx.loc[i, 'Bioactivity']
                                  
                                      predictions_list.append({'SMILES': smi, 'Predicted Activity': man, 'True Activity': true_value, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                   

                                   
                                    else:
                                    
                                     st.write(f" ### Give Header As Bioactivity . This is Classification Model")
                                 
                                  
                                    
                                    
                st.write(f" ### {f'p{content}'} value prediction")
                predictions_df = pd.DataFrame(predictions_list)
                if 'Predicted Value' in predictions_df.columns:
                 y_real = []
                 y_pred = []
                 y_pred.extend(predictions_df['Predicted Value'])
                 y_real.extend(predictions_df['True Value'])
                 y_real = np.array(y_real)
                 y_pred = np.array(y_pred)
                 r2 = r2_score(y_real, y_pred)
                 rmse = np.sqrt(mean_squared_error(y_real, y_pred))
                 st.write(predictions_df)
                 st.write(f" ### R2: {r2:.3f}")
                 st.write(f" ### RMSE: {rmse:.3f}")
                elif 'Predicted Activity' in predictions_df.columns:
                    y_real = []
                    y_pred = []
                    y_pred.extend(predictions_df['Predicted Activity'].map({'active': 1, 'inactive': 0}))
                    y_real.extend(predictions_df['True Activity'].map({'active': 1, 'inactive': 0}))
                    y_real = np.array(y_real)
                    y_pred = np.array(y_pred)
                    accuracy = accuracy_score(y_real, y_pred)

                    cm = confusion_matrix(y_real, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp)
                    sensitivity = tp / (tp + fn)
                    mcc = matthews_corrcoef(y_real, y_pred)
                    st.write(predictions_df)
                    st.write(f" ### Accuracy: {accuracy:.3f}")
                    st.write(f" ### Specificity: {specificity:.3f}")
                    st.write(f" ### Sensitivity: {sensitivity:.3f}")
                    st.write(f" ### MCC: {mcc:.3f}")
                  

               else:
                Smiles = dfx['SMILES']
                predictions_list = []
                for i, smi in enumerate(Smiles):
                     m = Chem.MolFromSmiles(smi, sanitize=False)
                     if m is None:
                          st.write(" ### Your **SMILES** is not correct  ")
                     else:
                        with open(task, 'r') as file:
                            con = file.read()
                            file.close()
                        if con == 'Regression':
                            xml_files = glob.glob(xml)
                            xml_files.sort() 
                            FP_list = ['AtomPairs2DCount',
                                'AtomPairs2D',
                                'EState',
                                'CDKextended',
                                'CDK',
                                'CDKgraphonly',
                                'KlekotaRothCount',
                                'KlekotaRoth',
                                'MACCS',
                                'PubChem',
                                'SubstructureCount',
                                'Substructure']
                            
                        
                            fp = dict(zip(FP_list, xml_files))
                            df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                            df.to_csv(smilefile, sep='\t', index=False, header=False)
                            with open(cfp, 'r') as file:
                                content = file.readline()
                            fingerprint = content
                            if fingerprint in FP_list:
                                fingerprint_descriptortypes = fp[fingerprint]
                                padeldescriptor(mol_dir= smilefile, 
                                            d_file=fingerprint_output_file, #'Substructure.csv'
                                            #descriptortypes='SubstructureFingerprint.xml', 
                                            descriptortypes= fingerprint_descriptortypes,
                                            detectaromaticity=True,
                                            standardizenitro=True,
                                            standardizetautomers=True,
                                            threads=2,
                                            removesalt=True,
                                            log=True,
                                            fingerprints=True)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Value'], axis=1)
                                fingerprintx = R.values
                                X = descriptors.drop(['Name'], axis=1)
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'
                                
                        
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                predx = round(float(pred[0]), 2)
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                
                                
                                
                                predictions_list.append({'SMILES': smi, 'Predicted Value': predx, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                

                            elif fingerprint == 'CSFP':
                        
                                df = pd.DataFrame({'SMILES': [smi], 'Name': [i]})
                                df.to_csv(smilefile, sep='\t', index=False, header=False)

                                smiles = df['SMILES']

                                # Load SMARTS patterns
                                SMARTS = np.loadtxt(smarts, dtype=str, comments=None)
                                CSFP = fingerprints.FragmentFingerprint(substructure_list=SMARTS.tolist())

                                # Transform SMILES to fingerprints
                                data_csfp = CSFP.transform_smiles(smiles)

                                # Convert sparse matrix to dense array
                                data_csfp_dense = data_csfp.toarray()

                                # Convert dense array to DataFrame
                                data_csfp_df = pd.DataFrame(data_csfp_dense)
                                
                                del data_csfp_df[data_csfp_df.columns[0]]

                                # Ensure there is no 'Unnamed: 0' column by setting index=False when saving
                                data_csfp_df.to_csv(fingerprint_output_file, index=False)

                                # Read descriptors from the CSV file
                                descriptors = pd.read_csv(fingerprint_output_file)

                                # Ensure no 'Unnamed: 0' column is expected in subsequent operations
                                

                                # Read X1 and drop 'Value' column
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Value'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]  
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'   
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                predx = round(float(pred[0]), 2)
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                
                              
                               
                                predictions_list.append({'SMILES': smi, 'Predicted Value': predx, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                            
                            elif fingerprint == 'RDKitDescriptors':
                                descriptor_names = [desc[0] for desc in Descriptors.descList]
                                calculator =MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                def compute_all_descriptors(smiles):
                                    descriptor_names = [desc[0] for desc in Descriptors.descList]
                                    calculator =MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        # Compute descriptors
                                        return pd.Series(calculator.CalcDescriptors(mol))
                                    else:
                                        # Return NaN for invalid SMILES
                                        return pd.Series([None] * len(descriptor_names))
                                df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                descriptors_df = df['SMILES'].apply(compute_all_descriptors)
                                descriptors_df = pd.DataFrame(descriptors_df.values.tolist(), columns=descriptor_names)
                                
                                nan_columns = descriptors_df.columns[descriptors_df.isna().any()].tolist()

                               

                                # Drop columns with NaN values
                                descriptors_df.drop(columns=nan_columns, inplace=True)
                                float32_max = np.finfo(np.float32).max
                                float32_min = np.finfo(np.float32).min
                                descriptors_df.replace([np.inf, -np.inf], [float32_max, float32_min], inplace=True)

                                # Clip values to fit float32 range
                                descriptors_df = descriptors_df.clip(lower=float32_min, upper=float32_max)

                                # Convert to float32
                                descriptors_df = descriptors_df.astype('float32')
                                
                                descriptors_df.to_csv(fingerprint_output_file,  index=False)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Value'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                dataxx = pd.read_excel(ADBoundpath, index_col=0)
                                dataxx = dataxx[R.columns]
                                min_bounds = dataxx.loc['Min']
                                max_bounds = dataxx.loc['Max']
                                compound_descriptors = X.iloc[0].to_dict()  # Assuming single compound; adjust if multiple
                                descriptor_results = {}
                                AD = 'IN Applicability Domain'
                                tolerance = 1e-6
                                for descriptor, value in compound_descriptors.items():
                                    if descriptor in min_bounds.index:
                                        is_within_range = min_bounds[descriptor] <= value <= max_bounds[descriptor]
                                        descriptor_results[descriptor] = "Within Range" if is_within_range else "Out of Range"

                                        # Update overall status if any descriptor is out of range
                                        if not is_within_range:
                                            AD = "OUT Applicability Domain"
                                    else:
                                        descriptor_results[descriptor] = "Descriptor not found in bounds data" 
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                predx = round(float(pred[0]), 2)
                              
                                with open(std, 'r') as file:
                                    content = file.read()
                                
                             
                                predictions_list.append({'SMILES': smi, 'Predicted Value': predx, 'AD': AD})
                                
                                   

                                   
                                

                              
                            else:
                                df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                smiles = df['SMILES']
                                compute_fingerprints(smiles, fingerprint)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Value'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'
                                
                              
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                predx = round(float(pred[0]), 2)
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                
                             
                               
                                predictions_list.append({'SMILES': smi, 'Predicted Value': predx, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                               
                        

                        else:
                            with open(cfp, 'r') as file:
                                    content = file.readline()
                            fingerprint = content
                            FP_list = ['AtomPairs2DCount',
                                    'AtomPairs2D',
                                    'EState',
                                    'CDKextended',
                                    'CDK',
                                    'CDKgraphonly',
                                    'KlekotaRothCount',
                                    'KlekotaRoth',
                                    'MACCS',
                                    'PubChem',
                                    'SubstructureCount',
                                    'Substructure']
                                
                            if fingerprint in FP_list:
                                xml_files = glob.glob(xml)
                                xml_files.sort() 
                                FP_list = ['AtomPairs2DCount',
                                    'AtomPairs2D',
                                    'EState',
                                    'CDKextended',
                                    'CDK',
                                    'CDKgraphonly',
                                    'KlekotaRothCount',
                                    'KlekotaRoth',
                                    'MACCS',
                                    'PubChem',
                                    'SubstructureCount',
                                    'Substructure']
                                
                            
                                fp = dict(zip(FP_list, xml_files))
                                df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                fingerprint_descriptortypes = fp[fingerprint]
                                padeldescriptor(mol_dir= smilefile, 
                                            d_file=fingerprint_output_file, #'Substructure.csv'
                                            #descriptortypes='SubstructureFingerprint.xml', 
                                            descriptortypes= fingerprint_descriptortypes,
                                            detectaromaticity=True,
                                            standardizenitro=True,
                                            standardizetautomers=True,
                                            threads=2,
                                            removesalt=True,
                                            log=True,
                                            fingerprints=True)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Bioactivity'], axis=1)
                                fingerprintx = R.values
                                X = descriptors.drop(['Name'], axis=1)
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'
                    
                                model = load(loadm)
                                
                                pred = model.predict(X)
                    
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                value = str(pred[0])
                                if value == '1':
                                    with open(below, 'r') as file:
                                        man = file.read()
                                  
                                  
                                    
                                    predictions_list.append({'SMILES': smi, 'Prediction': man, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                 
                                   
                            

                                else:
                                    with open(above, 'r') as file:
                                        man = file.read()
                                 
                                   
                                    
                                    predictions_list.append({'SMILES': smi, 'Prediction': man, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                 
                                  
                            elif fingerprint == 'CSFP':
                        
                                df = pd.DataFrame({'SMILES': [smi], 'Name': [i]})
                                df.to_csv(smilefile, sep='\t', index=False, header=False)

                                smiles = df['SMILES']

                                # Load SMARTS patterns
                                SMARTS = np.loadtxt(smarts, dtype=str, comments=None)
                                CSFP = fingerprints.FragmentFingerprint(substructure_list=SMARTS.tolist())

                                # Transform SMILES to fingerprints
                                data_csfp = CSFP.transform_smiles(smiles)

                                # Convert sparse matrix to dense array
                                data_csfp_dense = data_csfp.toarray()

                                # Convert dense array to DataFrame
                                data_csfp_df = pd.DataFrame(data_csfp_dense)
                                
                                del data_csfp_df[data_csfp_df.columns[0]]

                                # Ensure there is no 'Unnamed: 0' column by setting index=False when saving
                                data_csfp_df.to_csv(fingerprint_output_file, index=False)

                                # Read descriptors from the CSV file
                                descriptors = pd.read_csv(fingerprint_output_file)

                                # Ensure no 'Unnamed: 0' column is expected in subsequent operations
                                

                                # Read X1 and drop 'Value' column
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Bioactivity'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'
                                
                               
                                model = load(loadm)
                                
                                pred = model.predict(X)
                               
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                value = str(pred[0])
                                if value == '1':
                                    with open(below, 'r') as file:
                                        man = file.read()
                                   
                                
                                   
                                    predictions_list.append({'SMILES': smi, 'Prediction': man, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                 
                                   
                                else:
                                    with open(above, 'r') as file:
                                        man = file.read()
                                  
                                   
                                  
                                    predictions_list.append({'SMILES': smi, 'Prediction': man, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                 
                            elif fingerprint == 'RDKitDescriptors':
                                descriptor_names = [desc[0] for desc in Descriptors.descList]
                                calculator =MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                def compute_all_descriptors(smiles):
                                    descriptor_names = [desc[0] for desc in Descriptors.descList]
                                    calculator =MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        # Compute descriptors
                                        return pd.Series(calculator.CalcDescriptors(mol))
                                    else:
                                        # Return NaN for invalid SMILES
                                        return pd.Series([None] * len(descriptor_names))
                                df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                descriptors_df = df['SMILES'].apply(compute_all_descriptors)
                                descriptors_df = pd.DataFrame(descriptors_df.values.tolist(), columns=descriptor_names)
                                
                                nan_columns = descriptors_df.columns[descriptors_df.isna().any()].tolist()

                               

                                # Drop columns with NaN values
                                descriptors_df.drop(columns=nan_columns, inplace=True)
                                float32_max = np.finfo(np.float32).max
                                float32_min = np.finfo(np.float32).min
                                descriptors_df.replace([np.inf, -np.inf], [float32_max, float32_min], inplace=True)

                                # Clip values to fit float32 range
                                descriptors_df = descriptors_df.clip(lower=float32_min, upper=float32_max)

                                # Convert to float32
                                descriptors_df = descriptors_df.astype('float32')
                               
                                descriptors_df.to_csv(fingerprint_output_file,  index=False)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Bioactivity'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                dataxx = pd.read_excel(ADBoundpath, index_col=0)
                                dataxx = dataxx[R.columns]
                                min_bounds = dataxx.loc['Min']
                                max_bounds = dataxx.loc['Max']
                                compound_descriptors = X.iloc[0].to_dict()  # Assuming single compound; adjust if multiple
                                descriptor_results = {}
                                AD = "IN Applicability Domain"  # Default to "IN"
                                tolerance = 1e-6
                                for descriptor, value in compound_descriptors.items():
                                    if descriptor in min_bounds.index:
                                        is_within_range = min_bounds[descriptor] <= value <= max_bounds[descriptor]
                                        descriptor_results[descriptor] = "Within Range" if is_within_range else "Out of Range"

                                        # Update overall status if any descriptor is out of range
                                        if not is_within_range:
                                            AD = "OUT Applicability Domain"
                                    else:
                                        descriptor_results[descriptor] = "Descriptor not found in bounds data"     
                                model = load(loadm)
                                
                                pred = model.predict(X)
                           
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                value = str(pred[0])
                                if value == '1':
                                    with open(below, 'r') as file:
                                        man = file.read()
                                    
                                    predictions_list.append({'SMILES': smi, 'Prediction': man, 'AD': AD})
                                
                                  
                                   
                                else:
                                    with open(above, 'r') as file:
                                        man = file.read()
                                  


                                    predictions_list.append({'SMILES': smi, 'Prediction': man, 'AD': AD})
 
                            else:
                                df = pd.DataFrame({'SMILES': [smi], 'Name' : [i]} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                smiles = df['SMILES']
                                compute_fingerprints(smiles, fingerprint)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Bioactivity'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                external_molecule = np.array(fingerprinty)
                                tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                                sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                                top_k_indices = sorted_indices[1:6]
                                mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                                print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                                with open(TANPER, 'r') as file:
                                    content = file.read()
                                if mean_top_k_similarity >= AD_slider:
                                    AD = 'IN Applicability Domain'
                                else:
                                    AD = 'OUT Applicability Domain'
                                
                               
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                

                            
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                value = str(pred[0])
                                if value == '1':
                                    with open(below, 'r') as file:
                                        man = file.read()
                                    
                                   
                                    predictions_list.append({'SMILES': smi, 'Prediction': man, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                 
                                
                                else:
                                    with open(above, 'r') as file:
                                        man = file.read()
                                 
                                  
                                    
                                    predictions_list.append({'SMILES': smi, 'Prediction': man, 'AD': AD, 'Mean Tanimoto Similarity': mean_top_k_similarity})
                                 

                if con == "Regression":                  
                    st.write(f" ### {f'p{content}'} value prediction")
                    predictions_df = pd.DataFrame(predictions_list)
                    st.write(predictions_df)
                elif con == "Classification":
                    st.write(f" ### {f'p{content}'} value prediction")
                    predictions_df = pd.DataFrame(predictions_list)
                    st.write(predictions_df)
                  

                
                    

                    
                            
                 
    def model_pred():
        st.write(f" *The SMILES you wanted to predict activity* : {SMILES_input} ")
        if SMILES_input:
            m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

            
            if m is None:
                # The SMILES is invalid
                
                
                st.write(" ### Your **SMILES** is not correct  ")
            else:
                with open(task, 'r') as file:
                    con = file.read()
                    file.close()
                if con == 'Regression':
                    xml_files = glob.glob(xml)
                    xml_files.sort() 
                    FP_list = ['AtomPairs2DCount',
                        'AtomPairs2D',
                        'EState',
                        'CDKextended',
                        'CDK',
                        'CDKgraphonly',
                        'KlekotaRothCount',
                        'KlekotaRoth',
                        'MACCS',
                        'PubChem',
                        'SubstructureCount',
                        'Substructure']
                    
                   
                    fp = dict(zip(FP_list, xml_files))
                    df = pd.DataFrame({'SMILES': [SMILES_input], 'Name' : ['Molecule']} )
                    df.to_csv(smilefile, sep='\t', index=False, header=False)
                    with open(cfp, 'r') as file:
                        content = file.readline()
                    fingerprint = content
                    if fingerprint in FP_list:
                        fingerprint_descriptortypes = fp[fingerprint]
                        padeldescriptor(mol_dir= smilefile, 
                                    d_file=fingerprint_output_file, #'Substructure.csv'
                                    #descriptortypes='SubstructureFingerprint.xml', 
                                    descriptortypes= fingerprint_descriptortypes,
                                    detectaromaticity=True,
                                    standardizenitro=True,
                                    standardizetautomers=True,
                                    threads=2,
                                    removesalt=True,
                                    log=True,
                                    fingerprints=True)
                        descriptors = pd.read_csv(fingerprint_output_file)
                        X1 = pd.read_csv(xcol)
                        y = X1['Value'] 

                        R = X1.drop(['Value'], axis=1)
                        fingerprintx = R.values
                        X = descriptors.drop(['Name'], axis=1)
                        X = descriptors[R.columns]
                        fingerprinty = X.values
                        external_molecule = np.array(fingerprinty)
                        combined = np.vstack([fingerprintx, external_molecule])

                        # ---- TSNE ----
                        tsne = TSNE(
                            n_components=2,
                            perplexity=30,
                            learning_rate='auto',
                            init='pca',
                            random_state=42
                        )

                        embedding = tsne.fit_transform(combined)

                     
                        train_emb = embedding[:-1]
                        external_emb = embedding[-1]

                    
                        plt.figure(figsize=(8,6))

                        scatter = plt.scatter(
                            train_emb[:,0],
                            train_emb[:,1],
                            c=y,
                            cmap='viridis',
                            s=40,
                            alpha=0.8
                        )

                        # external molecule
                        plt.scatter(
                            external_emb[0],
                            external_emb[1],
                            color='red',
                            s=150,
                            marker='*',
                            label='Projected Molecule'
                        )

                        plt.colorbar(scatter,label='Value')
                        plt.title("Chemical Space (t-SNE)")
                        plt.xlabel("t-SNE 1")
                        plt.ylabel("t-SNE 2")
                        plt.legend()

                        plt.savefig("tsne_plot.png",dpi=300)
                        plt.close()
                        tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                        sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                        top_k_indices = sorted_indices[1:6]
                        mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                        print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)

                        with open(TANPER, 'r') as file:
                            content = file.read()
                        if mean_top_k_similarity >= AD_slider:
                            st.write(" ### AD: IN Applicability Domain")
                        else:
                            st.write("### AD: OUT Applicability Domain")
                        
                        st.write(' ### Molecular Fingerprint Of Your Structure')
                        st.write(X)
                        st.write(X.shape)
                        model = load(loadm)
                        
                        pred = model.predict(X)
                        predx = round(float(pred[0]), 2)  

                       
                    
                        with open(std, 'r') as file:
                            content = file.read()
                        
                        st.write(f" ### {f'p{content}'} value : {str(predx)}")
                        st.write(f" ### Mean Tanimoto Similarity of Top 5 Neighbors: {mean_top_k_similarity}")
                        st.write(" ### t-SNE Plot'")
                        st.image("tsne_plot.png")


                        if SMILES_input:
                            m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

                                
                            if m is None:
                                    # The SMILES is invalid
                                    
                                    
                                st.write(" ### Your **SMILES** is not correct  ")
                            else:
                                try:
                                            AllChem.Compute2DCoords(m)

                                            # Save the 2D structure as an image file
                                        
                                            img = Draw.MolToImage(m)
                                            img.save(smiledraw)
                                            st.image(smiledraw)
                                            # SMILES is valid, perform further processing
                                            st.write('### Molecular Properties')

                                            # Calculate Lipinski properties
                                            m = Chem.MolFromSmiles(SMILES_input)
                                            NHA = Lipinski.NumHAcceptors(m)
                                            NHD = Lipinski.NumHDonors(m)
                                            st.write(f'- **Number of Hydrogen Acceptor** : {NHA}')
                                            st.write(f'- **Number of Hydrogen Donors** : {NHD}')
                                            Molwt = Descriptors.ExactMolWt(m)
                                            Molwt = "{:.2f}".format(Molwt)
                                            st.write(f'- **Molecular Wieght** : {Molwt}')
                                            logP = Crippen.MolLogP(m)
                                            logP = "{:.2f}".format(logP)
                                            st.write(f'- **LogP** : {logP}')
                                            rb = Descriptors.NumRotatableBonds(m)
                                            st.write(f'- **Number of Rotatable Bonds** : {rb}')
                                            numatom = rdchem.Mol.GetNumAtoms(m)
                                            st.write(f'- **Number of Atoms** : {numatom}')
                                            mr = Crippen.MolMR(m)
                                            mr = "{:.2f}".format(mr)
                                            st.write(f'- **Molecular Refractivity** : {mr}')
                                            tsam = QED.properties(m).PSA
                                            st.write(f'- **Topology Polar Surface Area** : {tsam}')
                                            fc = rdmolops.GetFormalCharge(m)
                                            st.write(f'- **Formal Charge** : {fc}')
                                            ha = rdchem.Mol.GetNumHeavyAtoms(m)
                                            st.write(f'- **Number of Heavy Atoms** : {ha}')
                                            nr = rdMolDescriptors.CalcNumRings(m)
                                            st.write(f'- **Number of Rings** : {nr}')
                                            Lipin = "Pass" if (float(Molwt) <= 500 and float(logP) <= 5 and int(NHD) <= 5 and int(NHA) <= 10 and int(rb) <= 5) else "Fail"
                                            Ghose = "Pass" if (float(Molwt) >= 160 and float(Molwt) <= 480 and float(logP) >= -0.4 and float(logP) <= 5.6 and int(numatom) >= 20 and int(numatom) <= 70 and float(mr) >= 40 and float(mr) <= 130) else "Fail"
                                            veber = "Pass" if (int(rb) <= 10 and float(tsam) <= 140) else "Fail"
                                            Ruleof3 = "Pass" if (float(Molwt) <= 300 and float(logP) <= 3 and int(NHD) <= 3 and int(NHA) <= 3 and int(rb) <= 3) else "Fail"
                                            Reos = "Pass" if (float(Molwt) >= 200 and float(Molwt) <= 500 and float(logP) >= -5 and float(logP) <= 5 and int(NHD) >= 0 and int(NHD) <= 5 and int(NHA) >= 0 and int(NHA) <= 10 and int(fc) >= -2 and int(fc) <= 2 and int(rb) >= 0 and int(rb) <= 8 and int(ha) >= 15 and int(ha) <= 50) else "Fail"
                                            DrugLike = "Pass" if (float(Molwt) < 400 and int(nr) > 0 and int(rb) < 5 and int(NHD) <= 5 and int(NHA) <= 10 and float(logP) < 5) else "Fail"
                                            st.write('### Filter Profile')
                                            st.write(f"- **Lipinksi Filter** : {Lipin}")
                                            st.write(f"- **Ghose Filter** : {Ghose}")
                                            st.write(f"- **Veber Filter** : {veber}")
                                            st.write(f"- **Ruleof3** : {Ruleof3}")
                                            st.write(f"- **Reos Filter** : {Reos}")
                                            st.write(f"- **DrugLike Filter** : {DrugLike}")
                                            os.remove(smilefile)
                                            os.remove(fingerprint_output_file)
                                            os.remove(fingerprint_output_file_txt)

                                            # Generate 2D coordinates for the molecule
                                    
                                    
                                except Exception as e:
                                            # Handle exceptions during Lipinski property calculation or image saving
                                            error_message = "***SMILES***"
                    
                                            print(error_message)
                    elif fingerprint == 'CSFP':
                        
                        df = pd.DataFrame({'SMILES': [SMILES_input], 'Name': ['Molecule']})
                        df.to_csv(smilefile, sep='\t', index=False, header=False)

                        smiles = df['SMILES']

                        # Load SMARTS patterns
                        SMARTS = np.loadtxt(smarts, dtype=str, comments=None)
                        CSFP = fingerprints.FragmentFingerprint(substructure_list=SMARTS.tolist())

                        # Transform SMILES to fingerprints
                        data_csfp = CSFP.transform_smiles(smiles)

                        # Convert sparse matrix to dense array
                        data_csfp_dense = data_csfp.toarray()

                        # Convert dense array to DataFrame
                        data_csfp_df = pd.DataFrame(data_csfp_dense)
                        
                        del data_csfp_df[data_csfp_df.columns[0]]

                        # Ensure there is no 'Unnamed: 0' column by setting index=False when saving
                        data_csfp_df.to_csv(fingerprint_output_file, index=False)

                        # Read descriptors from the CSV file
                        descriptors = pd.read_csv(fingerprint_output_file)

                        # Ensure no 'Unnamed: 0' column is expected in subsequent operations
                        

                        # Read X1 and drop 'Value' column
                        X1 = pd.read_csv(xcol)
                        y = X1['Value']
                        R = X1.drop(['Value'], axis=1)
                        fingerprintx = R.values
                        X = descriptors[R.columns]
                        fingerprinty = X.values
                        external_molecule = np.array(fingerprinty)
                        combined = np.vstack([fingerprintx, external_molecule])

                        # ---- TSNE ----
                        tsne = TSNE(
                            n_components=2,
                            perplexity=30,
                            learning_rate='auto',
                            init='pca',
                            random_state=42
                        )

                        embedding = tsne.fit_transform(combined)

                     
                        train_emb = embedding[:-1]
                        external_emb = embedding[-1]

                    
                        plt.figure(figsize=(8,6))

                        scatter = plt.scatter(
                            train_emb[:,0],
                            train_emb[:,1],
                            c=y,
                            cmap='viridis',
                            s=40,
                            alpha=0.8
                        )

                        # external molecule
                        plt.scatter(
                            external_emb[0],
                            external_emb[1],
                            color='red',
                            s=150,
                            marker='*',
                            label='Projected Molecule'
                        )

                        plt.colorbar(scatter,label='Value')
                        plt.title("Chemical Space (t-SNE)")
                        plt.xlabel("t-SNE 1")
                        plt.ylabel("t-SNE 2")
                        plt.legend()

                        plt.savefig("tsne_plot.png",dpi=300)
                        plt.close()

                        tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                        sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                        top_k_indices = sorted_indices[1:6]
                        mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                        print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                        with open(TANPER, 'r') as file:
                            content = file.read()
                        if mean_top_k_similarity >= AD_slider:
                            st.write(" ### AD: IN Applicability Domain")
                        else:
                            st.write("### AD: OUT Applicability Domain")
                        
                        st.write(' ### Molecular Fingerprint Of Your Structure')
                        st.write(X)
                        st.write(X.shape)
                        model = load(loadm)
                        
                        pred = model.predict(X)
                        predx = round(float(pred[0]), 2)  
                     
                    
                        with open(std, 'r') as file:
                            content = file.read()
                        
                        st.write(f" ### {f'p{content}'} value : {str(predx)}")
                        st.write(f" ### Mean Tanimoto Similarity of Top 5 Neighbors: {mean_top_k_similarity}")
                        st.write(" ### t-SNE Plot")
                        st.image("tsne_plot.png")


                        if SMILES_input:
                            m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

                                
                            if m is None:
                                    # The SMILES is invalid
                                    
                                    
                                st.write(" ### Your **SMILES** is not correct  ")
                            else:
                                try:
                                            AllChem.Compute2DCoords(m)

                                            # Save the 2D structure as an image file
                                        
                                            img = Draw.MolToImage(m)
                                            img.save(smiledraw)
                                            st.image(smiledraw)
                                            # SMILES is valid, perform further processing
                                            st.write('### Molecular Properties')

                                            # Calculate Lipinski properties
                                            m = Chem.MolFromSmiles(SMILES_input)
                                            NHA = Lipinski.NumHAcceptors(m)
                                            NHD = Lipinski.NumHDonors(m)
                                            st.write(f'- **Number of Hydrogen Acceptor** : {NHA}')
                                            st.write(f'- **Number of Hydrogen Donors** : {NHD}')
                                            Molwt = Descriptors.ExactMolWt(m)
                                            Molwt = "{:.2f}".format(Molwt)
                                            st.write(f'- **Molecular Wieght** : {Molwt}')
                                            logP = Crippen.MolLogP(m)
                                            logP = "{:.2f}".format(logP)
                                            st.write(f'- **LogP** : {logP}')
                                            rb = Descriptors.NumRotatableBonds(m)
                                            st.write(f'- **Number of Rotatable Bonds** : {rb}')
                                            numatom = rdchem.Mol.GetNumAtoms(m)
                                            st.write(f'- **Number of Atoms** : {numatom}')
                                            mr = Crippen.MolMR(m)
                                            mr = "{:.2f}".format(mr)
                                            st.write(f'- **Molecular Refractivity** : {mr}')
                                            tsam = QED.properties(m).PSA
                                            st.write(f'- **Topology Polar Surface Area** : {tsam}')
                                            fc = rdmolops.GetFormalCharge(m)
                                            st.write(f'- **Formal Charge** : {fc}')
                                            ha = rdchem.Mol.GetNumHeavyAtoms(m)
                                            st.write(f'- **Number of Heavy Atoms** : {ha}')
                                            nr = rdMolDescriptors.CalcNumRings(m)
                                            st.write(f'- **Number of Rings** : {nr}')
                                            Lipin = "Pass" if (float(Molwt) <= 500 and float(logP) <= 5 and int(NHD) <= 5 and int(NHA) <= 10 and int(rb) <= 5) else "Fail"
                                            Ghose = "Pass" if (float(Molwt) >= 160 and float(Molwt) <= 480 and float(logP) >= -0.4 and float(logP) <= 5.6 and int(numatom) >= 20 and int(numatom) <= 70 and float(mr) >= 40 and float(mr) <= 130) else "Fail"
                                            veber = "Pass" if (int(rb) <= 10 and float(tsam) <= 140) else "Fail"
                                            Ruleof3 = "Pass" if (float(Molwt) <= 300 and float(logP) <= 3 and int(NHD) <= 3 and int(NHA) <= 3 and int(rb) <= 3) else "Fail"
                                            Reos = "Pass" if (float(Molwt) >= 200 and float(Molwt) <= 500 and float(logP) >= -5 and float(logP) <= 5 and int(NHD) >= 0 and int(NHD) <= 5 and int(NHA) >= 0 and int(NHA) <= 10 and int(fc) >= -2 and int(fc) <= 2 and int(rb) >= 0 and int(rb) <= 8 and int(ha) >= 15 and int(ha) <= 50) else "Fail"
                                            DrugLike = "Pass" if (float(Molwt) < 400 and int(nr) > 0 and int(rb) < 5 and int(NHD) <= 5 and int(NHA) <= 10 and float(logP) < 5) else "Fail"
                                            st.write('### Filter Profile')
                                            st.write(f"- **Lipinksi Filter** : {Lipin}")
                                            st.write(f"- **Ghose Filter** : {Ghose}")
                                            st.write(f"- **Veber Filter** : {veber}")
                                            st.write(f"- **Ruleof3** : {Ruleof3}")
                                            st.write(f"- **Reos Filter** : {Reos}")
                                            st.write(f"- **DrugLike Filter** : {DrugLike}")
                                            os.remove(smilefile)
                                            os.remove(fingerprint_output_file)
                                            os.remove(fingerprint_output_file_txt)

                                            # Generate 2D coordinates for the molecule
                                    
                                    
                                except Exception as e:
                                            # Handle exceptions during Lipinski property calculation or image saving
                                            error_message = "***SMILES***"
                                            print(error_message)
                    elif fingerprint == 'RDKitDescriptors':
                                descriptor_names = [desc[0] for desc in Descriptors.descList]
                                calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                def compute_all_descriptors(smiles):
                                    descriptor_names = [desc[0] for desc in Descriptors.descList]
                                    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        # Compute descriptors
                                        return pd.Series(calculator.CalcDescriptors(mol))
                                    else:
                                        # Return NaN for invalid SMILES
                                        return pd.Series([None] * len(descriptor_names))
                                df = pd.DataFrame({'SMILES': [SMILES_input], 'Name' : ['Molecule']} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                descriptors_df = df['SMILES'].apply(compute_all_descriptors)
                                descriptors_df = pd.DataFrame(descriptors_df.values.tolist(), columns=descriptor_names)
                                
                                nan_columns = descriptors_df.columns[descriptors_df.isna().any()].tolist()

                               

                                # Drop columns with NaN values
                                descriptors_df.drop(columns=nan_columns, inplace=True)
                                float32_max = np.finfo(np.float32).max
                                float32_min = np.finfo(np.float32).min
                                descriptors_df.replace([np.inf, -np.inf], [float32_max, float32_min], inplace=True)

                                # Clip values to fit float32 range
                                descriptors_df = descriptors_df.clip(lower=float32_min, upper=float32_max)

                                # Convert to float32
                                descriptors_df = descriptors_df.astype('float32')
                                descriptors_df.to_csv(fingerprint_output_file,  index=False)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Value'], axis=1)
                                fingerprintx = R.values
                                y = X1['Value']        # pIC50
                                X_train = fingerprintx
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                                dataxx = pd.read_excel(ADBoundpath, index_col=0)
                                dataxx = dataxx[R.columns]
                                min_bounds = dataxx.loc['Min']
                                max_bounds = dataxx.loc['Max']


                                compound_descriptors = X.iloc[0].to_dict()  # Assuming single compound; adjust if multiple
                                descriptor_results = {}
                                AD = "IN Applicability Domain"  # Default to "IN"
                                tolerance = 1e-6
                                for descriptor, value in compound_descriptors.items():
                                    if descriptor in min_bounds.index:
                                        is_within_range = min_bounds[descriptor]  <= value <= max_bounds[descriptor]
                                        descriptor_results[descriptor] = "Within Range" if is_within_range else "Out of Range"

                                        # Update overall status if any descriptor is out of range
                                        if not is_within_range:
                                            AD = "OUT Applicability Domain"
                                    else:
                                        descriptor_results[descriptor] = "Descriptor not found in bounds data"  

                                # Display the overall Applicability Domain status
                                st.write(f"### AD: {AD}")
                                model = load(loadm)
                                
                                pred = model.predict(X)
                                predx = round(float(pred[0]), 2)  
                              
                                with open(std, 'r') as file:
                                    content = file.read()

                                st.write(f" ### {f'p{content}'} value : {str(predx)}")
                            

                                if SMILES_input:
                                    m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

                                        
                                    if m is None:
                                            # The SMILES is invalid
                                            
                                            
                                        st.write(" ### Your **SMILES** is not correct  ")
                                    else:
                                        try:
                                            AllChem.Compute2DCoords(m)

                                            # Save the 2D structure as an image file
                                        
                                            img = Draw.MolToImage(m)
                                            img.save(smiledraw)
                                            st.image(smiledraw)
                                            # SMILES is valid, perform further processing
                                            st.write('### Molecular Properties')

                                            # Calculate Lipinski properties
                                            m = Chem.MolFromSmiles(SMILES_input)
                                            NHA = Lipinski.NumHAcceptors(m)
                                            NHD = Lipinski.NumHDonors(m)
                                            st.write(f'- **Number of Hydrogen Acceptor** : {NHA}')
                                            st.write(f'- **Number of Hydrogen Donors** : {NHD}')
                                            Molwt = Descriptors.ExactMolWt(m)
                                            Molwt = "{:.2f}".format(Molwt)
                                            st.write(f'- **Molecular Wieght** : {Molwt}')
                                            logP = Crippen.MolLogP(m)
                                            logP = "{:.2f}".format(logP)
                                            st.write(f'- **LogP** : {logP}')
                                            rb = Descriptors.NumRotatableBonds(m)
                                            st.write(f'- **Number of Rotatable Bonds** : {rb}')
                                            numatom = rdchem.Mol.GetNumAtoms(m)
                                            st.write(f'- **Number of Atoms** : {numatom}')
                                            mr = Crippen.MolMR(m)
                                            mr = "{:.2f}".format(mr)
                                            st.write(f'- **Molecular Refractivity** : {mr}')
                                            tsam = QED.properties(m).PSA
                                            st.write(f'- **Topology Polar Surface Area** : {tsam}')
                                            fc = rdmolops.GetFormalCharge(m)
                                            st.write(f'- **Formal Charge** : {fc}')
                                            ha = rdchem.Mol.GetNumHeavyAtoms(m)
                                            st.write(f'- **Number of Heavy Atoms** : {ha}')
                                            nr = rdMolDescriptors.CalcNumRings(m)
                                            st.write(f'- **Number of Rings** : {nr}')
                                            Lipin = "Pass" if (float(Molwt) <= 500 and float(logP) <= 5 and int(NHD) <= 5 and int(NHA) <= 10 and int(rb) <= 5) else "Fail"
                                            Ghose = "Pass" if (float(Molwt) >= 160 and float(Molwt) <= 480 and float(logP) >= -0.4 and float(logP) <= 5.6 and int(numatom) >= 20 and int(numatom) <= 70 and float(mr) >= 40 and float(mr) <= 130) else "Fail"
                                            veber = "Pass" if (int(rb) <= 10 and float(tsam) <= 140) else "Fail"
                                            Ruleof3 = "Pass" if (float(Molwt) <= 300 and float(logP) <= 3 and int(NHD) <= 3 and int(NHA) <= 3 and int(rb) <= 3) else "Fail"
                                            Reos = "Pass" if (float(Molwt) >= 200 and float(Molwt) <= 500 and float(logP) >= -5 and float(logP) <= 5 and int(NHD) >= 0 and int(NHD) <= 5 and int(NHA) >= 0 and int(NHA) <= 10 and int(fc) >= -2 and int(fc) <= 2 and int(rb) >= 0 and int(rb) <= 8 and int(ha) >= 15 and int(ha) <= 50) else "Fail"
                                            DrugLike = "Pass" if (float(Molwt) < 400 and int(nr) > 0 and int(rb) < 5 and int(NHD) <= 5 and int(NHA) <= 10 and float(logP) < 5) else "Fail"
                                            st.write('### Filter Profile')
                                            st.write(f"- **Lipinksi Filter** : {Lipin}")
                                            st.write(f"- **Ghose Filter** : {Ghose}")
                                            st.write(f"- **Veber Filter** : {veber}")
                                            st.write(f"- **Ruleof3** : {Ruleof3}")
                                            st.write(f"- **Reos Filter** : {Reos}")
                                            st.write(f"- **DrugLike Filter** : {DrugLike}")
                                            os.remove(smilefile)
                                            os.remove(fingerprint_output_file)
                                            os.remove(fingerprint_output_file_txt)

                                        except Exception as e:
                                            # Handle exceptions during Lipinski property calculation or image saving
                                            error_message = "***SMILES***"
                                            print(error_message)    # Generate 2D coordinates for the molecule
                                

                    else:
                        df = pd.DataFrame({'SMILES': [SMILES_input], 'Name' : ['Molecule']} )
                        df.to_csv(smilefile, sep='\t', index=False, header=False)
                        with open(cfp, 'r') as file:
                            content = file.readline()
                        fingerprint = content
                        smiles = df['SMILES']
                        if fingerprint == 'Morgan':
                                    molist, bitinfo = compute_fingerprints(smiles, fingerprint)
                                    mol = molist[0]
                                    bitInfo = bitinfo[0]
                                    nBits = 1024
                                    shap_per_bit = np.zeros(nBits, dtype=float)

                        else:
                                    compute_fingerprints(smiles, fingerprint)
                        descriptors = pd.read_csv(fingerprint_output_file)
                        X1 = pd.read_csv(xcol)
                        y = X1['Value'] 
                        R = X1.drop(['Value'], axis=1)
                        fingerprintx = R.values
                        
                        X = descriptors[R.columns]
                        fingerprinty = X.values
                        external_molecule = np.array(fingerprinty)
                        
                        combined = np.vstack([fingerprintx, external_molecule])

                        # ---- TSNE ----
                        tsne = TSNE(
                            n_components=2,
                            perplexity=30,
                            learning_rate='auto',
                            init='pca',
                            random_state=42
                        )

                        embedding = tsne.fit_transform(combined)

                     
                        train_emb = embedding[:-1]
                        external_emb = embedding[-1]

                    
                        plt.figure(figsize=(8,6))

                        scatter = plt.scatter(
                            train_emb[:,0],
                            train_emb[:,1],
                            c=y,
                            cmap='viridis',
                            s=40,
                            alpha=0.8
                        )

                        # external molecule
                        plt.scatter(
                            external_emb[0],
                            external_emb[1],
                            color='red',
                            s=150,
                            marker='*',
                            label='Projected Molecule'
                        )

                        plt.colorbar(scatter,label='Value')
                        plt.title("Chemical Space (t-SNE)")
                        plt.xlabel("t-SNE 1")
                        plt.ylabel("t-SNE 2")
                        plt.legend()

                        plt.savefig("tsne_plot.png",dpi=300)
                        plt.close()

                     
                        tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                        sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                        top_k_indices = sorted_indices[1:6]
                        mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                        print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                        with open(TANPER, 'r') as file:
                            content = file.read()
                        if mean_top_k_similarity >= AD_slider:
                            st.write(" ### AD: IN Applicability Domain")
                        else:
                            st.write("### AD: OUT Applicability Domain")
                        
                        st.write(' ### Molecular Fingerprint Of Your Structure')
                        st.write(X)
                        st.write(X.shape)
                        model = load(loadm)
                       
                        
                        pred = model.predict(X)
                        predx = round(float(pred[0]), 2)  
                        if fingerprint == 'Morgan' and (
                                    isinstance(model, RandomForestRegressor) or 
                                    isinstance(model, XGBRegressor)
                                ):
                                explainer = TreeExplainer(model)
                                shap_values = explainer.shap_values(X, check_additivity=False)[0]

                             
                                shap_per_bit = np.zeros(nBits)

                                for i, col in enumerate(X.columns):
                                    bit_index = int(col.split("_")[1]) - 1   # FP_1 → bit 0
                                    shap_per_bit[bit_index] = shap_values[i]

                               
                                atom_contrib = np.zeros(mol.GetNumAtoms())

                                for bit, value in enumerate(shap_per_bit):
                                    if bit in bitInfo:
                                        for atom_id, radius in bitInfo[bit]:
                                            atom_contrib[atom_id] += value

                              
                                cap = 0.55  # <= makes colors lighter (never full red/blue)

                                max_abs = max(abs(atom_contrib)) if max(abs(atom_contrib)) != 0 else 1.0

                                atom_colors = {}
                                highlight_atoms = []

                                for i, v in enumerate(atom_contrib):
                                    t = min(abs(v) / max_abs, cap)  # clamp
                                    if v > 0:
                                        # light pink -> moderate red (not full)
                                        atom_colors[i] = (1.0, 1.0 - t, 1.0 - t)
                                        highlight_atoms.append(i)
                                    elif v < 0:
                                        # light blue -> moderate blue (not full)
                                        atom_colors[i] = (1.0 - t, 1.0 - t, 1.0)
                                        highlight_atoms.append(i)

                             
                                drawer = rdMolDraw2D.MolDraw2DCairo(650, 450)
                               
                                drawer.DrawMolecule(
                                    mol,
                                    highlightAtoms=highlight_atoms,
                                    highlightAtomColors=atom_colors
                                )
                                drawer.FinishDrawing()

                                with open("shap_atom.png", "wb") as f:
                                    f.write(drawer.GetDrawingText())

                                st.write("### SHAP Atom Contributions")
                                st.write("Red atoms contribute positively to the prediction, while blue atoms contribute negatively. The intensity of the color indicates the strength of the contribution.")

                                st.image("shap_atom.png")
                       
                    
                        with open(std, 'r') as file:
                            content = file.read()
                        
                        st.write(f" ### {f'p{content}'} value : {str(predx)}")
                        st.write(f" ### Mean Tanimoto Similarity of Top 5 Neighbors: {mean_top_k_similarity}")
                        st.write("### t-SNE Plot ")
                        st.image("tsne_plot.png")


                        if SMILES_input:
                            m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

                                
                            if m is None:
                                    # The SMILES is invalid
                                    
                                    
                                st.write(" ### Your **SMILES** is not correct  ")
                            else:
                                try:
                                            AllChem.Compute2DCoords(m)

                                            # Save the 2D structure as an image file
                                        
                                            img = Draw.MolToImage(m)
                                            img.save(smiledraw)
                                            st.image(smiledraw)
                                            # SMILES is valid, perform further processing
                                            

                                            st.write('### Molecular Properties')

                                            # Calculate Lipinski properties
                                            m = Chem.MolFromSmiles(SMILES_input)
                                            NHA = Lipinski.NumHAcceptors(m)
                                            NHD = Lipinski.NumHDonors(m)
                                            st.write(f'- **Number of Hydrogen Acceptor** : {NHA}')
                                            st.write(f'- **Number of Hydrogen Donors** : {NHD}')
                                            Molwt = Descriptors.ExactMolWt(m)
                                            Molwt = "{:.2f}".format(Molwt)
                                            st.write(f'- **Molecular Wieght** : {Molwt}')
                                            logP = Crippen.MolLogP(m)
                                            logP = "{:.2f}".format(logP)
                                            st.write(f'- **LogP** : {logP}')
                                            rb = Descriptors.NumRotatableBonds(m)
                                            st.write(f'- **Number of Rotatable Bonds** : {rb}')
                                            numatom = rdchem.Mol.GetNumAtoms(m)
                                            st.write(f'- **Number of Atoms** : {numatom}')
                                            mr = Crippen.MolMR(m)
                                            mr = "{:.2f}".format(mr)
                                            st.write(f'- **Molecular Refractivity** : {mr}')
                                            tsam = QED.properties(m).PSA
                                            st.write(f'- **Topology Polar Surface Area** : {tsam}')
                                            fc = rdmolops.GetFormalCharge(m)
                                            st.write(f'- **Formal Charge** : {fc}')
                                            ha = rdchem.Mol.GetNumHeavyAtoms(m)
                                            st.write(f'- **Number of Heavy Atoms** : {ha}')
                                            nr = rdMolDescriptors.CalcNumRings(m)
                                            st.write(f'- **Number of Rings** : {nr}')
                                            Lipin = "Pass" if (float(Molwt) <= 500 and float(logP) <= 5 and int(NHD) <= 5 and int(NHA) <= 10 and int(rb) <= 5) else "Fail"
                                            Ghose = "Pass" if (float(Molwt) >= 160 and float(Molwt) <= 480 and float(logP) >= -0.4 and float(logP) <= 5.6 and int(numatom) >= 20 and int(numatom) <= 70 and float(mr) >= 40 and float(mr) <= 130) else "Fail"
                                            veber = "Pass" if (int(rb) <= 10 and float(tsam) <= 140) else "Fail"
                                            Ruleof3 = "Pass" if (float(Molwt) <= 300 and float(logP) <= 3 and int(NHD) <= 3 and int(NHA) <= 3 and int(rb) <= 3) else "Fail"
                                            Reos = "Pass" if (float(Molwt) >= 200 and float(Molwt) <= 500 and float(logP) >= -5 and float(logP) <= 5 and int(NHD) >= 0 and int(NHD) <= 5 and int(NHA) >= 0 and int(NHA) <= 10 and int(fc) >= -2 and int(fc) <= 2 and int(rb) >= 0 and int(rb) <= 8 and int(ha) >= 15 and int(ha) <= 50) else "Fail"
                                            DrugLike = "Pass" if (float(Molwt) < 400 and int(nr) > 0 and int(rb) < 5 and int(NHD) <= 5 and int(NHA) <= 10 and float(logP) < 5) else "Fail"
                                            st.write('### Filter Profile')
                                            st.write(f"- **Lipinksi Filter** : {Lipin}")
                                            st.write(f"- **Ghose Filter** : {Ghose}")
                                            st.write(f"- **Veber Filter** : {veber}")
                                            st.write(f"- **Ruleof3** : {Ruleof3}")
                                            st.write(f"- **Reos Filter** : {Reos}")
                                            st.write(f"- **DrugLike Filter** : {DrugLike}")
                                            os.remove(smilefile)
                                            os.remove(fingerprint_output_file)
                                            os.remove(fingerprint_output_file_txt)

                                            # Generate 2D coordinates for the molecule
                                    
                                    
                                except Exception as e:
                                            # Handle exceptions during Lipinski property calculation or image saving
                                            error_message = "***SMILES***"
                                            print(error_message)

                            
                            
                        

                else:
                  with open(cfp, 'r') as file:
                        content = file.readline()
                  fingerprint = content
                  FP_list = ['AtomPairs2DCount',
                        'AtomPairs2D',
                        'EState',
                        'CDKextended',
                        'CDK',
                        'CDKgraphonly',
                        'KlekotaRothCount',
                        'KlekotaRoth',
                        'MACCS',
                        'PubChem',
                        'SubstructureCount',
                        'Substructure']
                    
                  if fingerprint in FP_list:
                    xml_files = glob.glob(xml)
                    xml_files.sort() 
                    FP_list = ['AtomPairs2DCount',
                        'AtomPairs2D',
                        'EState',
                        'CDKextended',
                        'CDK',
                        'CDKgraphonly',
                        'KlekotaRothCount',
                        'KlekotaRoth',
                        'MACCS',
                        'PubChem',
                        'SubstructureCount',
                        'Substructure']
                    
                
                    fp = dict(zip(FP_list, xml_files))
                    df = pd.DataFrame({'SMILES': [SMILES_input], 'Name' : ['Molecule']} )
                    df.to_csv(smilefile, sep='\t', index=False, header=False)
                    with open(cfp, 'r') as file:
                        content = file.readline()
                    fingerprint = content
                    fingerprint_descriptortypes = fp[fingerprint]
                    padeldescriptor(mol_dir= smilefile, 
                                d_file=fingerprint_output_file, #'Substructure.csv'
                                #descriptortypes='SubstructureFingerprint.xml', 
                                descriptortypes= fingerprint_descriptortypes,
                                detectaromaticity=True,
                                standardizenitro=True,
                                standardizetautomers=True,
                                threads=2,
                                removesalt=True,
                                log=True,
                                fingerprints=True)
                    descriptors = pd.read_csv(fingerprint_output_file)
                    X1 = pd.read_csv(xcol)
                    y = X1['Bioactivity']
                    R = X1.drop(['Bioactivity'], axis=1)
                    fingerprintx = R.values
                    X = descriptors.drop(['Name'], axis=1)
                    X = descriptors[R.columns]
                    fingerprinty = X.values
                    external_molecule = np.array(fingerprinty)
                    combined = np.vstack([fingerprintx, external_molecule])

                 
                    tsne = TSNE(
                        n_components=2,
                        perplexity=30,
                        learning_rate="auto",
                        init="pca",
                        random_state=42
                    )

                    embedding = tsne.fit_transform(combined)

                    train_emb = embedding[:-1]
                    external_emb = embedding[-1]

                    # ---- plot ----
                    plt.figure(figsize=(8, 6))

                    # Use discrete colormap for class labels
                    scatter = plt.scatter(
                        train_emb[:, 0],
                        train_emb[:, 1],
                        c=y,
                        cmap="coolwarm",     # blue/red
                        s=40,
                        alpha=0.85
                    )

                 
                    plt.scatter(
                        external_emb[0],
                        external_emb[1],
                        color="black",
                        s=180,
                        marker="*",
                        label="Projected Molecule"
                    )

                    # colorbar ticks for classes
                    cbar = plt.colorbar(scatter, ticks=[0, 1])
                    cbar.set_label("Bioactivity")
                    cbar.set_ticklabels(["Inactive (0)", "Active (1)"])

                    plt.title("Chemical Space (t-SNE) - Classification")
                    plt.xlabel("t-SNE 1")
                    plt.ylabel("t-SNE 2")
                    plt.legend()

                    plt.savefig("tsne_classification.png", dpi=300)
                    plt.close()

                   
                    tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                    sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                    top_k_indices = sorted_indices[1:6]
                    mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                    print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                    with open(TANPER, 'r') as file:
                        content = file.read()
                    if mean_top_k_similarity >= AD_slider:
                        st.write(" ### AD: IN Applicability Domain")
                    else:
                        st.write("### AD: OUT Applicability Domain")
                    
                    st.write(' ### Molecular Fingerprint Of Your Structure')
                    st.write(X)
                    st.write(X.shape)
                    model = load(loadm)
                    
                    pred = model.predict(X)
                 
                    
                   
                    with open(std, 'r') as file:
                        content = file.read()
                    value = str(pred[0])
                    if value == '1':
                     with open(below, 'r') as file:
                        man = file.read()
                     st.write(f" ### {f'{content}'} bioactivity of given molecule is : {man}")
                     st.write(f" ### Mean Tanimoto Similarity of Top 5 Neighbors: {mean_top_k_similarity}")
                     st.write("### t-SNE Plot ")
                     st.image("tsne_classification.png")
                   
                    else:
                         with open(above, 'r') as file:
                              man = file.read()
                         st.write(f" ### {f'{content}'} bioactivity of given molecule is : {man}")
                         st.write(f" ### Mean Tanimoto Similarity of Top 5 Neighbors: {mean_top_k_similarity}")
                         st.write("### t-SNE Plot ")
                         st.image("tsne_classification.png")
                       



                    if SMILES_input:
                        m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

                            
                        if m is None:
                                # The SMILES is invalid
                                
                                
                            st.write(" ### Your **SMILES** is not correct  ")
                        else:
                            try:
                                        AllChem.Compute2DCoords(m)

                                        # Save the 2D structure as an image file
                                    
                                        img = Draw.MolToImage(m)
                                        img.save(smiledraw)
                                        st.image(smiledraw)
                                        # SMILES is valid, perform further processing
                                    

                                        st.write('### Molecular Properties')

                                        # Calculate Lipinski properties
                                        m = Chem.MolFromSmiles(SMILES_input)
                                        NHA = Lipinski.NumHAcceptors(m)
                                        NHD = Lipinski.NumHDonors(m)
                                        st.write(f'- **Number of Hydrogen Acceptor** : {NHA}')
                                        st.write(f'- **Number of Hydrogen Donors** : {NHD}')
                                        Molwt = Descriptors.ExactMolWt(m)
                                        Molwt = "{:.2f}".format(Molwt)
                                        st.write(f'- **Molecular Wieght** : {Molwt}')
                                        logP = Crippen.MolLogP(m)
                                        logP = "{:.2f}".format(logP)
                                        st.write(f'- **LogP** : {logP}')
                                        rb = Descriptors.NumRotatableBonds(m)
                                        st.write(f'- **Number of Rotatable Bonds** : {rb}')
                                        numatom = rdchem.Mol.GetNumAtoms(m)
                                        st.write(f'- **Number of Atoms** : {numatom}')
                                        mr = Crippen.MolMR(m)
                                        mr = "{:.2f}".format(mr)
                                        st.write(f'- **Molecular Refractivity** : {mr}')
                                        tsam = QED.properties(m).PSA
                                        st.write(f'- **Topology Polar Surface Area** : {tsam}')
                                        fc = rdmolops.GetFormalCharge(m)
                                        st.write(f'- **Formal Charge** : {fc}')
                                        ha = rdchem.Mol.GetNumHeavyAtoms(m)
                                        st.write(f'- **Number of Heavy Atoms** : {ha}')
                                        nr = rdMolDescriptors.CalcNumRings(m)
                                        st.write(f'- **Number of Rings** : {nr}')
                                        Lipin = "Pass" if (float(Molwt) <= 500 and float(logP) <= 5 and int(NHD) <= 5 and int(NHA) <= 10 and int(rb) <= 5) else "Fail"
                                        Ghose = "Pass" if (float(Molwt) >= 160 and float(Molwt) <= 480 and float(logP) >= -0.4 and float(logP) <= 5.6 and int(numatom) >= 20 and int(numatom) <= 70 and float(mr) >= 40 and float(mr) <= 130) else "Fail"
                                        veber = "Pass" if (int(rb) <= 10 and float(tsam) <= 140) else "Fail"
                                        Ruleof3 = "Pass" if (float(Molwt) <= 300 and float(logP) <= 3 and int(NHD) <= 3 and int(NHA) <= 3 and int(rb) <= 3) else "Fail"
                                        Reos = "Pass" if (float(Molwt) >= 200 and float(Molwt) <= 500 and float(logP) >= -5 and float(logP) <= 5 and int(NHD) >= 0 and int(NHD) <= 5 and int(NHA) >= 0 and int(NHA) <= 10 and int(fc) >= -2 and int(fc) <= 2 and int(rb) >= 0 and int(rb) <= 8 and int(ha) >= 15 and int(ha) <= 50) else "Fail"
                                        DrugLike = "Pass" if (float(Molwt) < 400 and int(nr) > 0 and int(rb) < 5 and int(NHD) <= 5 and int(NHA) <= 10 and float(logP) < 5) else "Fail"
                                        st.write('### Filter Profile')
                                        st.write(f"- **Lipinksi Filter** : {Lipin}")
                                        st.write(f"- **Ghose Filter** : {Ghose}")
                                        st.write(f"- **Veber Filter** : {veber}")
                                        st.write(f"- **Ruleof3** : {Ruleof3}")
                                        st.write(f"- **Reos Filter** : {Reos}")
                                        st.write(f"- **DrugLike Filter** : {DrugLike}")
                                        os.remove(smilefile)
                                        os.remove(fingerprint_output_file)
                                        os.remove(fingerprint_output_file_txt)

                                        # Generate 2D coordinates for the molecule
                                
                                    
                            except Exception as e:
                                        # Handle exceptions during Lipinski property calculation or image saving
                                        error_message = "***SMILES***"
                                        print(error_message)
                    else:
                        # SMILES value is empty or doesn't exist
                            
                            
                        st.write(' ## SMILES not given')
                  elif fingerprint == 'CSFP':
                        
                        df = pd.DataFrame({'SMILES': [SMILES_input], 'Name': ['Molecule']})
                        df.to_csv(smilefile, sep='\t', index=False, header=False)

                        smiles = df['SMILES']

                        # Load SMARTS patterns
                        SMARTS = np.loadtxt(smarts, dtype=str, comments=None)
                        CSFP = fingerprints.FragmentFingerprint(substructure_list=SMARTS.tolist())

                        # Transform SMILES to fingerprints
                        data_csfp = CSFP.transform_smiles(smiles)

                        # Convert sparse matrix to dense array
                        data_csfp_dense = data_csfp.toarray()

                        # Convert dense array to DataFrame
                        data_csfp_df = pd.DataFrame(data_csfp_dense)
                        
                        del data_csfp_df[data_csfp_df.columns[0]]

                        # Ensure there is no 'Unnamed: 0' column by setting index=False when saving
                        data_csfp_df.to_csv(fingerprint_output_file, index=False)

                        # Read descriptors from the CSV file
                        descriptors = pd.read_csv(fingerprint_output_file)

                        # Ensure no 'Unnamed: 0' column is expected in subsequent operations
                        

                        # Read X1 and drop 'Value' column
                        X1 = pd.read_csv(xcol)
                        y = X1['Bioactivity']
                        R = X1.drop(['Bioactivity'], axis=1)
                        fingerprintx = R.values
                        
                        X = descriptors[R.columns]
                        fingerprinty = X.values
                        external_molecule = np.array(fingerprinty)
                        combined = np.vstack([fingerprintx, external_molecule])
                        tsne = TSNE(
                        n_components=2,
                        perplexity=30,
                        learning_rate="auto",
                        init="pca",
                        random_state=42
                    )

                        embedding = tsne.fit_transform(combined)

                        train_emb = embedding[:-1]
                        external_emb = embedding[-1]

                        # ---- plot ----
                        plt.figure(figsize=(8, 6))

                        # Use discrete colormap for class labels
                        scatter = plt.scatter(
                            train_emb[:, 0],
                            train_emb[:, 1],
                            c=y,
                            cmap="coolwarm",     # blue/red
                            s=40,
                            alpha=0.85
                        )

                    
                        plt.scatter(
                            external_emb[0],
                            external_emb[1],
                            color="black",
                            s=180,
                            marker="*",
                            label="Projected Molecule"
                        )

                        # colorbar ticks for classes
                        cbar = plt.colorbar(scatter, ticks=[0, 1])
                        cbar.set_label("Bioactivity")
                        cbar.set_ticklabels(["Inactive (0)", "Active (1)"])

                        plt.title("Chemical Space (t-SNE) - Classification")
                        plt.xlabel("t-SNE 1")
                        plt.ylabel("t-SNE 2")
                        plt.legend()

                        plt.savefig("tsne_classification.png", dpi=300)
                        plt.close()

                        tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                        sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                        top_k_indices = sorted_indices[1:6]
                        mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                        print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                        with open(TANPER, 'r') as file:
                            content = file.read()
                        if mean_top_k_similarity >= AD_slider:
                            st.write(" ### AD: IN Applicability Domain")
                        else:
                            st.write("### AD: OUT Applicability Domain")
                        
                        st.write(' ### Molecular Fingerprint Of Your Structure')
                        st.write(X)
                        st.write(X.shape)
                        model = load(loadm)
                        
                        pred = model.predict(X)
                     
                    
                        with open(std, 'r') as file:
                            content = file.read()
                        value = str(pred[0])
                        if value == '1':
                         with open(below, 'r') as file:
                            man = file.read()
                         st.write(f" ### {f'{content}'} bioactivity of given molecule is : {man}")
                         st.write(f" ### Mean Tanimoto Similarity of Top 5 Neighbors: {mean_top_k_similarity}")
                         st.write("### t-SNE Plot ")
                         st.image("tsne_classification.png")
                     
                         
                        else:
                            with open(above, 'r') as file:
                                man = file.read()
                            st.write(f" ### {f'{content}'} bioactivity of given molecule is : {man}")
                            st.write(f" ### Mean Tanimoto Similarity of Top 5 Neighbors: {mean_top_k_similarity}")
                            st.write("### t-SNE Plot ")
                            st.image("tsne_classification.png")
                           



                        if SMILES_input:
                            m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

                                
                            if m is None:
                                    # The SMILES is invalid
                                    
                                    
                                st.write(" ### Your **SMILES** is not correct  ")
                            else:
                                try:
                                            AllChem.Compute2DCoords(m)

                                            # Save the 2D structure as an image file
                                        
                                            img = Draw.MolToImage(m)
                                            img.save(smiledraw)
                                            st.image(smiledraw)
                                            # SMILES is valid, perform further processing
                                         
                                            st.write('### Molecular Properties')

                                            # Calculate Lipinski properties
                                            m = Chem.MolFromSmiles(SMILES_input)
                                            NHA = Lipinski.NumHAcceptors(m)
                                            NHD = Lipinski.NumHDonors(m)
                                            st.write(f'- **Number of Hydrogen Acceptor** : {NHA}')
                                            st.write(f'- **Number of Hydrogen Donors** : {NHD}')
                                            Molwt = Descriptors.ExactMolWt(m)
                                            Molwt = "{:.2f}".format(Molwt)
                                            st.write(f'- **Molecular Wieght** : {Molwt}')
                                            logP = Crippen.MolLogP(m)
                                            logP = "{:.2f}".format(logP)
                                            st.write(f'- **LogP** : {logP}')
                                            rb = Descriptors.NumRotatableBonds(m)
                                            st.write(f'- **Number of Rotatable Bonds** : {rb}')
                                            numatom = rdchem.Mol.GetNumAtoms(m)
                                            st.write(f'- **Number of Atoms** : {numatom}')
                                            mr = Crippen.MolMR(m)
                                            mr = "{:.2f}".format(mr)
                                            st.write(f'- **Molecular Refractivity** : {mr}')
                                            tsam = QED.properties(m).PSA
                                            st.write(f'- **Topology Polar Surface Area** : {tsam}')
                                            fc = rdmolops.GetFormalCharge(m)
                                            st.write(f'- **Formal Charge** : {fc}')
                                            ha = rdchem.Mol.GetNumHeavyAtoms(m)
                                            st.write(f'- **Number of Heavy Atoms** : {ha}')
                                            nr = rdMolDescriptors.CalcNumRings(m)
                                            st.write(f'- **Number of Rings** : {nr}')
                                            Lipin = "Pass" if (float(Molwt) <= 500 and float(logP) <= 5 and int(NHD) <= 5 and int(NHA) <= 10 and int(rb) <= 5) else "Fail"
                                            Ghose = "Pass" if (float(Molwt) >= 160 and float(Molwt) <= 480 and float(logP) >= -0.4 and float(logP) <= 5.6 and int(numatom) >= 20 and int(numatom) <= 70 and float(mr) >= 40 and float(mr) <= 130) else "Fail"
                                            veber = "Pass" if (int(rb) <= 10 and float(tsam) <= 140) else "Fail"
                                            Ruleof3 = "Pass" if (float(Molwt) <= 300 and float(logP) <= 3 and int(NHD) <= 3 and int(NHA) <= 3 and int(rb) <= 3) else "Fail"
                                            Reos = "Pass" if (float(Molwt) >= 200 and float(Molwt) <= 500 and float(logP) >= -5 and float(logP) <= 5 and int(NHD) >= 0 and int(NHD) <= 5 and int(NHA) >= 0 and int(NHA) <= 10 and int(fc) >= -2 and int(fc) <= 2 and int(rb) >= 0 and int(rb) <= 8 and int(ha) >= 15 and int(ha) <= 50) else "Fail"
                                            DrugLike = "Pass" if (float(Molwt) < 400 and int(nr) > 0 and int(rb) < 5 and int(NHD) <= 5 and int(NHA) <= 10 and float(logP) < 5) else "Fail"
                                            st.write('### Filter Profile')
                                            st.write(f"- **Lipinksi Filter** : {Lipin}")
                                            st.write(f"- **Ghose Filter** : {Ghose}")
                                            st.write(f"- **Veber Filter** : {veber}")
                                            st.write(f"- **Ruleof3** : {Ruleof3}")
                                            st.write(f"- **Reos Filter** : {Reos}")
                                            st.write(f"- **DrugLike Filter** : {DrugLike}")
                                            os.remove(smilefile)
                                            os.remove(fingerprint_output_file)
                                            os.remove(fingerprint_output_file_txt)

                                            # Generate 2D coordinates for the molecule
                                    
                                    
                                except Exception as e:
                                            # Handle exceptions during Lipinski property calculation or image saving
                                            error_message = "***SMILES***"
                                            print(error_message)  

                  elif fingerprint == 'RDKitDescriptors':
                                descriptor_names = [desc[0] for desc in Descriptors.descList]
                                calculator =MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                def compute_all_descriptors(smiles):
                                    descriptor_names = [desc[0] for desc in Descriptors.descList]
                                    calculator =MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        # Compute descriptors
                                        return pd.Series(calculator.CalcDescriptors(mol))
                                    else:
                                        # Return NaN for invalid SMILES
                                        return pd.Series([None] * len(descriptor_names))
                                df = pd.DataFrame({'SMILES': [SMILES_input], 'Name' : ['Molecule']} )
                                df.to_csv(smilefile, sep='\t', index=False, header=False)
                                with open(cfp, 'r') as file:
                                    content = file.readline()
                                fingerprint = content
                                descriptors_df = df['SMILES'].apply(compute_all_descriptors)
                                descriptors_df = pd.DataFrame(descriptors_df.values.tolist(), columns=descriptor_names)
                                
                                nan_columns = descriptors_df.columns[descriptors_df.isna().any()].tolist()

                               

                                # Drop columns with NaN values
                                descriptors_df.drop(columns=nan_columns, inplace=True)
                                float32_max = np.finfo(np.float32).max
                                float32_min = np.finfo(np.float32).min
                                descriptors_df.replace([np.inf, -np.inf], [float32_max, float32_min], inplace=True)

                                # Clip values to fit float32 range
                                descriptors_df = descriptors_df.clip(lower=float32_min, upper=float32_max)

                                # Convert to float32
                                descriptors_df = descriptors_df.astype('float32')
                                descriptors_df.to_csv(fingerprint_output_file,  index=False)
                                descriptors = pd.read_csv(fingerprint_output_file)
                                X1 = pd.read_csv(xcol)
                                R = X1.drop(['Bioactivity'], axis=1)
                                fingerprintx = R.values
                                X = descriptors[R.columns]
                                fingerprinty = X.values
                               

                             
                                dataxx = pd.read_excel(ADBoundpath, index_col=0)
                                dataxx = dataxx[R.columns]
                                min_bounds = dataxx.loc['Min']
                                max_bounds = dataxx.loc['Max']
                                compound_descriptors = X.iloc[0].to_dict()  # Assuming single compound; adjust if multiple
                                descriptor_results = {}
                                AD = "IN Applicability Domain"  # Default to "IN"
                                tolerance = 1e-6
                                for descriptor, value in compound_descriptors.items():
                                    if descriptor in min_bounds.index:
                                        is_within_range = min_bounds[descriptor] <= value <= max_bounds[descriptor] 
                                        descriptor_results[descriptor] = "Within Range" if is_within_range else "Out of Range"

                                        # Update overall status if any descriptor is out of range
                                        if not is_within_range:
                                            AD = "OUT Applicability Domain"
                                    else:
                                        descriptor_results[descriptor] = "Descriptor not found in bounds data"  
                                # Display the overall Applicability Domain status
                                st.write(f"### AD: {AD}")
                                   
                                model = load(loadm)
                                
                                pred = model.predict(X)
                           
                            
                                with open(std, 'r') as file:
                                    content = file.read()
                                value = str(pred[0])
                                if value == '1':
                                    with open(below, 'r') as file:
                                        man = file.read()
                                    
                                    st.write(f" ### {f'{content}'} bioactivity of given molecule is : {man}")
                                  
                                
                                  
                                   
                                elif value == '0':
                                    with open(above, 'r') as file:
                                        man = file.read()
                                  


                                    st.write(f" ### {f'{content}'} bioactivity of given molecule is : {man}")
                                  
                                if SMILES_input:
                                    m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

                                        
                                    if m is None:
                                            # The SMILES is invalid
                                            
                                            
                                        st.write(" ### Your **SMILES** is not correct  ")
                                    else:
                                     try:
                                                AllChem.Compute2DCoords(m)

                                                # Save the 2D structure as an image file
                                            
                                                img = Draw.MolToImage(m)
                                                img.save(smiledraw)
                                                st.image(smiledraw)
                                                # SMILES is valid, perform further processing
                                            
                                                st.write('### Molecular Properties')

                                                # Calculate Lipinski properties
                                                m = Chem.MolFromSmiles(SMILES_input)
                                                NHA = Lipinski.NumHAcceptors(m)
                                                NHD = Lipinski.NumHDonors(m)
                                                st.write(f'- **Number of Hydrogen Acceptor** : {NHA}')
                                                st.write(f'- **Number of Hydrogen Donors** : {NHD}')
                                                Molwt = Descriptors.ExactMolWt(m)
                                                Molwt = "{:.2f}".format(Molwt)
                                                st.write(f'- **Molecular Wieght** : {Molwt}')
                                                logP = Crippen.MolLogP(m)
                                                logP = "{:.2f}".format(logP)
                                                st.write(f'- **LogP** : {logP}')
                                                rb = Descriptors.NumRotatableBonds(m)
                                                st.write(f'- **Number of Rotatable Bonds** : {rb}')
                                                numatom = rdchem.Mol.GetNumAtoms(m)
                                                st.write(f'- **Number of Atoms** : {numatom}')
                                                mr = Crippen.MolMR(m)
                                                mr = "{:.2f}".format(mr)
                                                st.write(f'- **Molecular Refractivity** : {mr}')
                                                tsam = QED.properties(m).PSA
                                                st.write(f'- **Topology Polar Surface Area** : {tsam}')
                                                fc = rdmolops.GetFormalCharge(m)
                                                st.write(f'- **Formal Charge** : {fc}')
                                                ha = rdchem.Mol.GetNumHeavyAtoms(m)
                                                st.write(f'- **Number of Heavy Atoms** : {ha}')
                                                nr = rdMolDescriptors.CalcNumRings(m)
                                                st.write(f'- **Number of Rings** : {nr}')
                                                Lipin = "Pass" if (float(Molwt) <= 500 and float(logP) <= 5 and int(NHD) <= 5 and int(NHA) <= 10 and int(rb) <= 5) else "Fail"
                                                Ghose = "Pass" if (float(Molwt) >= 160 and float(Molwt) <= 480 and float(logP) >= -0.4 and float(logP) <= 5.6 and int(numatom) >= 20 and int(numatom) <= 70 and float(mr) >= 40 and float(mr) <= 130) else "Fail"
                                                veber = "Pass" if (int(rb) <= 10 and float(tsam) <= 140) else "Fail"
                                                Ruleof3 = "Pass" if (float(Molwt) <= 300 and float(logP) <= 3 and int(NHD) <= 3 and int(NHA) <= 3 and int(rb) <= 3) else "Fail"
                                                Reos = "Pass" if (float(Molwt) >= 200 and float(Molwt) <= 500 and float(logP) >= -5 and float(logP) <= 5 and int(NHD) >= 0 and int(NHD) <= 5 and int(NHA) >= 0 and int(NHA) <= 10 and int(fc) >= -2 and int(fc) <= 2 and int(rb) >= 0 and int(rb) <= 8 and int(ha) >= 15 and int(ha) <= 50) else "Fail"
                                                DrugLike = "Pass" if (float(Molwt) < 400 and int(nr) > 0 and int(rb) < 5 and int(NHD) <= 5 and int(NHA) <= 10 and float(logP) < 5) else "Fail"
                                                st.write('### Filter Profile')
                                                st.write(f"- **Lipinksi Filter** : {Lipin}")
                                                st.write(f"- **Ghose Filter** : {Ghose}")
                                                st.write(f"- **Veber Filter** : {veber}")
                                                st.write(f"- **Ruleof3** : {Ruleof3}")
                                                st.write(f"- **Reos Filter** : {Reos}")
                                                st.write(f"- **DrugLike Filter** : {DrugLike}")
                                                os.remove(smilefile)
                                                os.remove(fingerprint_output_file)
                                                os.remove(fingerprint_output_file_txt)

                                                # Generate 2D coordinates for the molecule
                                        
                                        
                                     except Exception as e:
                                                # Handle exceptions during Lipinski property calculation or image saving
                                                error_message = "***SMILES***"
                                                print(error_message)

  
            
                  else:
                        df = pd.DataFrame({'SMILES': [SMILES_input], 'Name' : ['Molecule']} )
                        df.to_csv(smilefile, sep='\t', index=False, header=False)
                        with open(cfp, 'r') as file:
                            content = file.readline()
                        fingerprint = content
                        smiles = df['SMILES']
                        if fingerprint == 'Morgan':
                                    molist, bitinfo = compute_fingerprints(smiles, fingerprint)
                                    mol = molist[0]
                                    bitInfo = bitinfo[0]
                                    nBits = 1024
                                    shap_per_bit = np.zeros(nBits, dtype=float)

                        else:
                                    compute_fingerprints(smiles, fingerprint)
                        descriptors = pd.read_csv(fingerprint_output_file)
                        X1 = pd.read_csv(xcol)
                        y = X1['Bioactivity']
                        R = X1.drop(['Bioactivity'], axis=1)
                        fingerprintx = R.values
                        X = descriptors[R.columns]
                        fingerprinty = X.values
                        external_molecule = np.array(fingerprinty)
                        combined = np.vstack([fingerprintx, external_molecule])
                        tsne = TSNE(
                        n_components=2,
                        perplexity=30,
                        learning_rate="auto",
                        init="pca",
                        random_state=42
                        )

                        embedding = tsne.fit_transform(combined)

                        train_emb = embedding[:-1]
                        external_emb = embedding[-1]

                        # ---- plot ----
                        plt.figure(figsize=(8, 6))

                        # Use discrete colormap for class labels
                        scatter = plt.scatter(
                            train_emb[:, 0],
                            train_emb[:, 1],
                            c=y,
                            cmap="coolwarm",     # blue/red
                            s=40,
                            alpha=0.85
                        )

                    
                        plt.scatter(
                            external_emb[0],
                            external_emb[1],
                            color="black",
                            s=180,
                            marker="*",
                            label="Projected Molecule"
                        )

                        # colorbar ticks for classes
                        cbar = plt.colorbar(scatter, ticks=[0, 1])
                        cbar.set_label("Bioactivity")
                        cbar.set_ticklabels(["Inactive (0)", "Active (1)"])

                        plt.title("Chemical Space (t-SNE) - Classification")
                        plt.xlabel("t-SNE 1")
                        plt.ylabel("t-SNE 2")
                        plt.legend()

                        plt.savefig("tsne_classification.png", dpi=300)
                        plt.close()

                        tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
                        sorted_indices = np.argsort(tanimoto_similarities)[::-1]
                        top_k_indices = sorted_indices[1:6]
                        mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
                        print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
                        with open(TANPER, 'r') as file:
                            content = file.read()
                        if mean_top_k_similarity >= AD_slider:
                            st.write(" ### AD: IN Applicability Domain")
                        else:
                            st.write("### AD: OUT Applicability Domain")
                        
                        st.write(' ### Molecular Fingerprint Of Your Structure')
                        st.write(X)
                        st.write(X.shape)
                        model = load(loadm)
                        
                        pred = model.predict(X)
                       
                       
                     
                      
                    
                        with open(std, 'r') as file:
                            content = file.read()
                        value = str(pred[0])
                        if value == '1':
                         with open(below, 'r') as file:
                            man = file.read()
                         st.write(f" ### {f'{content}'} bioactivity of given molecule is : {man}")
                         st.write(f" ### Mean Tanimoto Similarity of Top 5 Neighbors: {mean_top_k_similarity}")
                         st.write("### t-SNE Plot ")
                         st.image("tsne_classification.png")
                        
                        
                        else:
                            with open(above, 'r') as file:
                                man = file.read()
                            st.write(f" ### {f'{content}'} bioactivity of given molecule is : {man}")
                            st.write(f" ### Mean Tanimoto Similarity of Top 5 Neighbors: {mean_top_k_similarity}")
                            st.write("### t-SNE Plot ")
                            st.image("tsne_classification.png")
                         


                        if SMILES_input:
                            m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

                                
                            if m is None:
                                    # The SMILES is invalid
                                    
                                    
                                st.write(" ### Your **SMILES** is not correct  ")
                            else:
                                try:
                                            AllChem.Compute2DCoords(m)

                                            # Save the 2D structure as an image file
                                        
                                            img = Draw.MolToImage(m)
                                            img.save(smiledraw)
                                            st.image(smiledraw)
                                            # SMILES is valid, perform further processing
                                         
                                            st.write('### Molecular Properties')

                                            # Calculate Lipinski properties
                                            m = Chem.MolFromSmiles(SMILES_input)
                                            NHA = Lipinski.NumHAcceptors(m)
                                            NHD = Lipinski.NumHDonors(m)
                                            st.write(f'- **Number of Hydrogen Acceptor** : {NHA}')
                                            st.write(f'- **Number of Hydrogen Donors** : {NHD}')
                                            Molwt = Descriptors.ExactMolWt(m)
                                            Molwt = "{:.2f}".format(Molwt)
                                            st.write(f'- **Molecular Wieght** : {Molwt}')
                                            logP = Crippen.MolLogP(m)
                                            logP = "{:.2f}".format(logP)
                                            st.write(f'- **LogP** : {logP}')
                                            rb = Descriptors.NumRotatableBonds(m)
                                            st.write(f'- **Number of Rotatable Bonds** : {rb}')
                                            numatom = rdchem.Mol.GetNumAtoms(m)
                                            st.write(f'- **Number of Atoms** : {numatom}')
                                            mr = Crippen.MolMR(m)
                                            mr = "{:.2f}".format(mr)
                                            st.write(f'- **Molecular Refractivity** : {mr}')
                                            tsam = QED.properties(m).PSA
                                            st.write(f'- **Topology Polar Surface Area** : {tsam}')
                                            fc = rdmolops.GetFormalCharge(m)
                                            st.write(f'- **Formal Charge** : {fc}')
                                            ha = rdchem.Mol.GetNumHeavyAtoms(m)
                                            st.write(f'- **Number of Heavy Atoms** : {ha}')
                                            nr = rdMolDescriptors.CalcNumRings(m)
                                            st.write(f'- **Number of Rings** : {nr}')
                                            Lipin = "Pass" if (float(Molwt) <= 500 and float(logP) <= 5 and int(NHD) <= 5 and int(NHA) <= 10 and int(rb) <= 5) else "Fail"
                                            Ghose = "Pass" if (float(Molwt) >= 160 and float(Molwt) <= 480 and float(logP) >= -0.4 and float(logP) <= 5.6 and int(numatom) >= 20 and int(numatom) <= 70 and float(mr) >= 40 and float(mr) <= 130) else "Fail"
                                            veber = "Pass" if (int(rb) <= 10 and float(tsam) <= 140) else "Fail"
                                            Ruleof3 = "Pass" if (float(Molwt) <= 300 and float(logP) <= 3 and int(NHD) <= 3 and int(NHA) <= 3 and int(rb) <= 3) else "Fail"
                                            Reos = "Pass" if (float(Molwt) >= 200 and float(Molwt) <= 500 and float(logP) >= -5 and float(logP) <= 5 and int(NHD) >= 0 and int(NHD) <= 5 and int(NHA) >= 0 and int(NHA) <= 10 and int(fc) >= -2 and int(fc) <= 2 and int(rb) >= 0 and int(rb) <= 8 and int(ha) >= 15 and int(ha) <= 50) else "Fail"
                                            DrugLike = "Pass" if (float(Molwt) < 400 and int(nr) > 0 and int(rb) < 5 and int(NHD) <= 5 and int(NHA) <= 10 and float(logP) < 5) else "Fail"
                                            st.write('### Filter Profile')
                                            st.write(f"- **Lipinksi Filter** : {Lipin}")
                                            st.write(f"- **Ghose Filter** : {Ghose}")
                                            st.write(f"- **Veber Filter** : {veber}")
                                            st.write(f"- **Ruleof3** : {Ruleof3}")
                                            st.write(f"- **Reos Filter** : {Reos}")
                                            st.write(f"- **DrugLike Filter** : {DrugLike}")
                                            os.remove(smilefile)
                                            os.remove(fingerprint_output_file)
                                            os.remove(fingerprint_output_file_txt)

                                            # Generate 2D coordinates for the molecule
                                    
                                    
                                except Exception as e:
                                            # Handle exceptions during Lipinski property calculation or image saving
                                            error_message = "***SMILES***"
                                            print(error_message)




    if button:
        model_pred()
    if button2:
         model_predexcel()
    
with tab1:
    with open(about, 'r') as file:
        content = file.read()

    st.write(content)
    
with tab2:
    with open(data, 'r') as file:
        content = file.read()
    st.write(content)
    with open(task,'r') as file:
        contentx = file.read()
    if contentx == 'Regression':
        with open(cfp, 'r') as file:
          content = file.readline()
        fingerprint = content
        if fingerprint == 'RDKitDescriptors':
            st.image(dataimg, caption= 'Scatter Plot of y_real and y_pred')
           
        else:
            st.image(dataimg, caption= 'Scatter Plot of y_real and y_pred')
            st.image(TPNG, caption= 'The Tanimoto Similarity Data Distribution:-')
            
    else:
        with open(cfp, 'r') as file:
          content = file.readline()
        fingerprint = content
        if fingerprint == 'RDKitDescriptors':
            st.image(dataimg, caption= 'Classification Metrics of the Model')
           
        else:
            st.image(dataimg, caption= 'Classification Metrics of the Model')
            st.image(TPNG, caption= 'The Tanimoto Similarity Data Distribution:-')
            

        

with tab3:
    with open(model, 'r') as file:
        content = file.read()
        

    st.write(content)
    st.image(image)

with tab4:
    with open(citation, 'r') as file:
        content = file.read()
    st.write(content)
    with open(citationlink, 'r') as file:
        content = file.read()
    st.link_button('Go To Article Page', url=f'{content}')

with tab5:
    with open(author, 'r') as file:
        content = file.read()
    
    
    st.write(content)