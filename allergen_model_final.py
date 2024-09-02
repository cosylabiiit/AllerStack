# Uncomment the following lines to install necessary packages if running in a new environment
#!pip install pandas scikit-learn lazypredict
#!pip install pandas scikit-learn shap matplotlib
# !pip3 install Biopython
# !pip install torch esm biopython tensorflow==2.12.0 scikeras==0.10.0 keras==2.15.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
import pickle
import Bio
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import torch
import esm
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tqdm import tqdm

# Dataset input
def input_allergen():
    sequence = input("\nEnter the sequence \t\t\t\t")
    return sequence

# Data preprocess
# Define function to check for non-natural amino acids
def has_non_natural(seq):
    non_natural_amino_acids = {'B', 'O', 'J', 'U', 'X', 'Z'}
    return any(aa in non_natural_amino_acids for aa in seq)

# Features
def calculate_aac(sequence):
    analysed_seq = ProteinAnalysis(sequence)
    aac = analysed_seq.get_amino_acids_percent()
    return aac

def calculate_dipeptide_composition(sequence):
    dipeptides = [a+b for a in 'ACDEFGHIKLMNPQRSTVWY' for b in 'ACDEFGHIKLMNPQRSTVWY']
    composition = {dipeptide: 0 for dipeptide in dipeptides}
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if dipeptide in composition:
            composition[dipeptide] += 1
    total = sum(composition.values())
    composition = {k: v / total for k, v in composition.items()}
    return composition

def calculate_pseudo_aac(sequence, lamda=10, weight=0.05):
    analysed_seq = ProteinAnalysis(sequence)
    aac = analysed_seq.get_amino_acids_percent()
    theta = [analysed_seq.molecular_weight() / (i + 1) for i in range(lamda)]
    pseudo_aac = {f"PAAC_{i+1}": weight * theta[i] for i in range(lamda)}
    pseudo_aac.update(aac)
    return pseudo_aac

def calculate_physicochemical_properties(sequence):
    analysed_seq = ProteinAnalysis(sequence)
    properties = {
        "Molecular_Weight": analysed_seq.molecular_weight(),
        "Aromaticity": analysed_seq.aromaticity(),
        "Instability_Index": analysed_seq.instability_index(),
        "Isoelectric_Point": analysed_seq.isoelectric_point(),
        "Secondary_Structure_Fraction": analysed_seq.secondary_structure_fraction()
    }
    return properties

def extract_features(input_file):
    output_file = input_file.rstrip('.csv') + "_features.csv"
    df_in = pd.read_csv(input_file)
    sequences = df_in['Sequence'].tolist()

    features = []
    for seq in sequences:
        feature_set = {}
        feature_set.update(calculate_aac(seq))
        feature_set.update(calculate_dipeptide_composition(seq))
        feature_set.update(calculate_pseudo_aac(seq))
        feature_set.update(calculate_physicochemical_properties(seq))
        features.append(feature_set)

    features_df = pd.DataFrame(features)
    result_df = pd.concat([df_in, features_df], axis=1)
    result_df.to_csv(output_file, index=False)

    return result_df, output_file

# ESM2 feature extraction function
def extract_esm2_features(df, batch_size=50):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disable dropout for deterministic results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sequences = df['Sequence'].tolist()
    sequence_ids = df['Sequence_ID'].tolist()
    data = list(zip(sequence_ids, sequences))

    sequence_representations = []

    for start in tqdm(range(0, len(data), batch_size), desc="Processing ESM Batches"):
        end = start + batch_size
        batch_data = data[start:end]

        try:
            batch_labels, batch_tokens, batch_lengths = batch_converter(batch_data)
            batch_tokens = torch.tensor(batch_tokens).to(device)
        except Exception as e:
            print(f"Error in batch_converter: {e}")
            continue

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
            token_embeddings = results["representations"][33]

        for i, (seq_id, seq) in enumerate(batch_data):
            seq_embedding = token_embeddings[i, 1:len(seq) + 1].mean(0).cpu().numpy()
            sequence_representations.append(seq_embedding)

    df['esm_embeddings'] = sequence_representations
    return df

# Feature selection
def selected_features(df):
    selected_features_main = [
    'feature_2', 'feature_3', 'feature_10', 'feature_18', 'feature_26',
    'feature_27', 'feature_38', 'feature_57', 'feature_65', 'feature_68',
    'feature_90', 'feature_91', 'feature_101', 'feature_113', 'feature_119',
    'feature_141', 'feature_149', 'feature_162', 'feature_163', 'feature_171',
    'feature_173', 'feature_186', 'feature_190', 'feature_195', 'feature_197',
    'feature_210', 'feature_222', 'feature_225', 'feature_230', 'feature_234',
    'feature_235', 'feature_254', 'feature_265', 'feature_271', 'feature_276',
    'feature_298', 'feature_313', 'feature_317', 'feature_322', 'feature_333',
    'feature_356', 'feature_365', 'feature_382', 'feature_383', 'feature_396',
    'feature_404', 'feature_414', 'feature_417', 'feature_440', 'feature_482',
    'feature_492', 'feature_497', 'feature_515', 'feature_520', 'feature_526',
    'feature_534', 'feature_546', 'feature_565', 'feature_568', 'feature_615',
    'feature_643', 'feature_652', 'feature_653', 'feature_654', 'feature_657',
    'feature_660', 'feature_690', 'feature_697', 'feature_704', 'feature_717',
    'feature_719', 'feature_744', 'feature_784', 'feature_793', 'feature_800',
    'feature_802', 'feature_805', 'feature_818', 'feature_830', 'feature_841',
    'feature_842', 'feature_843', 'feature_853', 'feature_855', 'feature_858',
    'feature_860', 'feature_869', 'feature_876', 'feature_878', 'feature_880',
    'feature_895', 'feature_896', 'feature_914', 'feature_941', 'feature_943',
    'feature_946', 'feature_979', 'feature_984', 'feature_988', 'feature_989',
    'feature_992', 'feature_1003', 'feature_1007', 'feature_1013', 'feature_1015',
    'feature_1025', 'feature_1031', 'feature_1048', 'feature_1051', 'feature_1065',
    'feature_1069', 'feature_1072', 'feature_1077', 'feature_1085', 'feature_1093',
    'feature_1112', 'feature_1119', 'feature_1122', 'feature_1144', 'feature_1145',
    'feature_1149', 'feature_1158', 'feature_1162', 'feature_1163', 'feature_1182',
    'feature_1185', 'feature_1213', 'feature_1223', 'feature_1228', 'feature_1233',
    'feature_1245', 'feature_1246', 'feature_1248'
]
    selected_features_qda = [
    'C', 'D', 'E', 'K', 'W', 'AA', 'AC', 'AE', 'AH', 'AT', 'CC', 'CG', 'CK',
    'CN', 'DK', 'EC', 'EE', 'EQ', 'FE', 'FT', 'GA', 'GQ', 'GW', 'HF', 'HP',
    'IE', 'IK', 'IN', 'KD', 'LE', 'LS', 'MG', 'ML', 'NW', 'NY', 'PP', 'RL',
    'RV', 'ST', 'SV', 'VT', 'WC', 'WD', 'YF', 'YG', 'YR', 'YY', 'PAAC_5',
    'PAAC_9', 'Aromaticity', 'Isoelectric_Point',
    'Secondary_Structure_Fraction1', 'Secondary_Structure_Fraction3']

    df_selected_feature_main = df[selected_features_main]
    df_selected_features_qda = df[selected_features_qda]

    return df_selected_feature_main, df_selected_features_qda

# Model prediction function
def model(select_feat_main, select_feat_qda):
    with open('qda_model.pkl', 'rb') as file:
        base1 = pickle.load(file)

    with open('svc_model.pkl', 'rb') as file:
        base2 = pickle.load(file)

    with open('knn_model.pkl', 'rb') as file:
        base3 = pickle.load(file)

    with open('ann_model.pkl', 'rb') as file:
        base4 = pickle.load(file)

    meta1 = base1.predict(select_feat_qda)
    meta2 = base2.predict(select_feat_main)
    meta3 = base3.predict(select_feat_main)
    meta4 = base4.predict(select_feat_main)

    meta_feat = np.column_stack([meta1, meta2, meta3, meta4])

    with open('meta_model.pkl', 'rb') as file:
        meta = pickle.load(file)

    final_prediction = meta.predict(meta_feat)
    return final_prediction

# Main function
def main():
    input_seq = input_allergen()
    df = pd.DataFrame({"Sequence": [input_seq]})
    df['Sequence'] = df['Sequence'].str.replace('\n', '').str.replace(' ', '')

    contains_non_natural = df['Sequence'].apply(has_non_natural)
    df = df[~contains_non_natural]

    df, output_file = extract_features(df)
    df = extract_esm2_features(df)

    df[['Secondary_Structure_Fraction1', 'Secondary_Structure_Fraction2', 'Secondary_Structure_Fraction3']] = df['Secondary_Structure_Fraction'].str.strip('()').str.split(',', expand=True).astype(float)
    df = df.drop(columns=['Secondary_Structure_Fraction'])

    select_feat_main, select_feat_qda = selected_features(df)
    prediction_result = model(select_feat_main, select_feat_qda)

    print(prediction_result)

# Execute main function if this script is run directly
if __name__ == "__main__":
    main()