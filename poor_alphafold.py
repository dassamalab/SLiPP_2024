from Bio.PDB import *
import json


def extract_confidence_score(structure):
    total_atoms = Selection.unfold_entities(structure, 'A')
    confidence_score = 0
    for atom in total_atoms:
        confidence_score += atom.get_bfactor()
    confidence_score = confidence_score / len(total_atoms)
    return confidence_score


def extract_length(structure):
    total_residues = Selection.unfold_entities(structure, 'R')
    return len(total_residues)


def remove_signal_peptide(pdb_file, cleave_site):
    parser = PDBParser()
    try:
        structure = parser.get_structure('prot', pdb_file)
        Dice.extract(structure, 'A', int(cleave_site), int(extract_length(structure)), pdb_file)
    except:
        print('error')


def batch_remove_signal_peptide(organism, signalp_json):
    with open(signalp_json) as f:
        result = json.load(f)
        result = result['SEQUENCES']
        for key in result:
            cleave_site = result[key]['CS_pos']
            if cleave_site != '':
                cleave_site = cleave_site[34:36]
                if key[3] == 'A':
                    uniprot_id = key[3:13]
                else:
                    uniprot_id = key[3:9]
                pdb_file = f'{organism}/{uniprot_id}.pdb'
                remove_signal_peptide(pdb_file, cleave_site)


def main():
    remove_signal_peptide('Bburgdorferi/G5IXG9.pdb', 20)


if __name__ == '__main__':
    main()
