import pandas as pd
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.metrics
from biobb_vs.fpocket.fpocket_run import fpocket_run
import os
from Bio.PDB import *
from pathlib import Path
import json
import slipp_utils
import argparse

NAME_CONVERSION = {'volume': 'pock_vol', 'number_of_alpha_spheres': 'nb_AS', 'mean_alpha_sphere_radius': 'mean_as_ray',
                   'mean_alp_sph_solvent_access': 'mean_as_solv_acc', 'apolar_alpha_sphere_proportion': 'apol_as_prop',
                   'mean_local_hydrophobic_density': 'mean_loc_hyd_dens', 'hydrophobicity_score': 'hydrophobicity_score',
                   'volume_score': 'volume_score', 'polarity_score': 'polarity_score', 'charge_score': 'charge_score',
                   'flexibility': 'flex', 'proportion_of_polar_atoms': 'prop_polar_atm', 'alpha_sphere_density': 'as_density',
                   'cent_of_mass_alpha_sphere_max_dist': 'as_max_dst', 'polar_sasa': 'surf_pol_vdw14',
                   'apolar_sasa': 'surf_apol_vdw14', 'total_sasa': 'surf_vdw14'}
SELECTED_PARAM = ['pock_vol', 'nb_AS', 'mean_as_ray', 'mean_as_solv_acc', 'apol_as_prop', 'mean_loc_hyd_dens',
                  'hydrophobicity_score', 'volume_score', 'polarity_score', 'charge_score', 'flex', 'prop_polar_atm',
                  'as_density', 'as_max_dst', 'surf_pol_vdw14', 'surf_apol_vdw14', 'surf_vdw14']


def download_pdb(file):
    """
    Given a txt file containing a list of pdb ids, download the pdb files and save them as .pdb files in the existing
    folder.
    :param file: the text file path containing a list of pdb ids
    :return folder_name: as string to indicate the directory for input and output
    :return pdb_ids: as a list of strings having all the pdb ids
    """
    folder_name = file.split('.')[0]
    with open(file) as f:
        pdb_ids = f.readlines()
    for i in range(len(pdb_ids)):
        pdb_ids[i] = pdb_ids[i].strip()
    item = PDBList()
    item.download_pdb_files(pdb_codes=pdb_ids, pdir=f'{folder_name}/', file_format='pdb')
    for pdb in pdb_ids:     # resave all the files into .pdb files to be compatible with other functions
        try:
            native_pose = Path(f'{folder_name}/pdb{pdb}.ent')
            native_pose.rename(native_pose.with_suffix('.pdb'))
        except:
            continue
    return folder_name, pdb_ids


def retrieve_alphafold_structure(file):
    """
    Given a txt file containing a list of UniProt ids, download the AlphaFold .pdb files and save them in the existing
    folder.
    :param file: the text file path containing a list of UniProt ids
    :return folder_name: as string to indicate the directory for input and output
    :return pdb_ids: as a list of strings having all the UniProt ids
    """
    folder_name = file.split('.')[0]
    with open(file) as f:
        accession_ids = f.readlines()
    for i in range(len(accession_ids)):
        accession_ids[i] = accession_ids[i].strip()
        # could change v3 to other numbers to retrieve the latest AF model
        model_url = f'https://alphafold.ebi.ac.uk/files/AF-{accession_ids[i]}-F1-model_v3.pdb'
        os.system(f'curl {model_url} -o {folder_name}/{accession_ids[i]}.pdb')
    return folder_name, accession_ids


def generate_fpocket_results_pdb(folder_name, code):
    fpocket_run(f'{folder_name}/pdb{code}.pdb', f'{folder_name}/pdb{code}.zip', f'{folder_name}/pdb{code}.json')


def generate_fpocket_results_alphafold(folder_name, code):
    fpocket_run(f'{folder_name}/{code}.pdb', f'{folder_name}/{code}.zip', f'{folder_name}/pdb{code}.json')


def export_parameters_from_fpocket(folder_name, code):
    """
    Output all the fpocket predicted pockets into a pandas dataframe given the ids
    :param folder_name: the folder name where the fpocket results and pdb files are stored, normally would be species
                        name
    :param code: the pdb or UniProt ID to retrieve pocket information
    :return: a pandas dataframe storing all the pockets predicted by fpocket
    """
    with open(f'{folder_name}/pdb{code}.json') as f:
        data = json.load(f)     # json imported as dictionary
        all_poc_info = pd.DataFrame()
        for i in range(1, len(data)+1):
            all_poc_info = all_poc_info.append(data[f'pocket{i}'], ignore_index=True)
    return all_poc_info


def training(total_df):
    """
    Create the random forest model with the given training dataset.
    :param total_df: a pandas dataframe with training dataset.
    :return: a random forest classifier to predict probabilities of pockets being lipid binding pockets.
    """
    x_total = total_df[SELECTED_PARAM]
    y_total = total_df.lipid_binding
    rfc = sklearn.ensemble.RandomForestClassifier()
    rfc.fit(x_total, y_total)
    return rfc


def batch_predict(model, file, num_json_files):
    """
    Given the pre-trained random forest model and the file containing all the id codes, generate a pandas dataframe
    containing all the prediction result.
    :param model: the already trained random forest model
    :param file: a .txt file with all the id codes representing the proteins to be predicted
    :param num_json_files: int that represents the number of signalP prediction result files
    :return: a dataframe that contains the prediction result
    """
    with open(file) as f:
        ids = f.readlines()
    result = pd.DataFrame(columns=['id_code'] + SELECTED_PARAM)
    if len(ids[0].strip()) == 4:
        folder_name, ids = download_pdb(file)
        for id_code in ids:
            generate_fpocket_results_pdb(folder_name, id_code)
            raw_poc_info = export_parameters_from_fpocket(folder_name, id_code)
            raw_poc_info = raw_poc_info.rename(columns=NAME_CONVERSION)
            predicted_poc_info = predict(model, raw_poc_info)
            best_prediction_i = predicted_poc_info.idxmax(axis=0)['prediction']
            predicted_poc_info['id_code'] = id_code
            poc_info = predicted_poc_info.iloc[best_prediction_i]
            result = result.append(poc_info, ignore_index=True)
    else:
        folder_name, ids = retrieve_alphafold_structure(file)
        if num_json_files:
            for i in range(num_json_files):
                slipp_utils.batch_remove_signal_peptide(folder_name, f'{folder_name}_{i}.json')
        for id_code in ids:
            parser = PDBParser()
            try:
                structure = parser.get_structure('prot', f'{folder_name}/{id_code}.pdb')
            except:
                continue
            if slipp_utils.extract_length(structure) > 100:
                if slipp_utils.extract_confidence_score(structure) > 70:
                    generate_fpocket_results_alphafold(folder_name, id_code)
                    raw_poc_info = export_parameters_from_fpocket(folder_name, id_code)
                    raw_poc_info = raw_poc_info.rename(columns=NAME_CONVERSION)
                    if not raw_poc_info.empty:
                        predicted_poc_info = predict(model, raw_poc_info)
                        best_prediction_i = predicted_poc_info.idxmax(axis=0)['prediction']
                        predicted_poc_info['id_code'] = id_code
                        poc_info = predicted_poc_info.iloc[best_prediction_i]
                    else:
                        poc_info = {'id_code': id_code, 'error': 'No pocket found'}
                else:
                    poc_info = {'id_code': id_code, 'error': 'Poor prediction'}
            else:
                poc_info = {'id_code': id_code, 'error': 'Too short'}
            result = result.append(poc_info, ignore_index=True)
    return result


def predict(model, dataframe):
    prediction = model.predict_proba(dataframe[SELECTED_PARAM])
    dataframe['prediction'] = prediction[:, 1]
    return dataframe


def main():
    parser = argparse.ArgumentParser(prog='SLiPP', description='Predict lipid binding proteins from given structures')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='the filename of the input file containing the protein ids')
    parser.add_argument('-s', '--signalpeptide', default=0, required=False, type=int,
                        help='the number of signalP prediction files')
    parser.add_argument('-o', '--output', required=True, type=str, help='the output filename')

    total_df = pd.read_csv('training_pockets.csv')
    rfc = training(total_df)
    args = parser.parse_args()
    fpocket_info = batch_predict(rfc, args.input, args.signalpeptide)
    fpocket_info.to_csv(args.output)


if __name__ == '__main__':
    main()
