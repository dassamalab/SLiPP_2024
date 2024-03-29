# Structure-based Lipid-interacting pocket predictor (SLiPP)
The program extracts pocket information from given protein models and predict the possibility of the pocket being lipid-binding pockets through machine learning.
## Requirments
- Python 3.2+
- [fpocket](https://github.com/Discngine/fpocket)
- Used python packages: pandas v1.1.2, scikit-learn v0.23.2, biobb_vs v4.0.0, biopython v1.79
## Usage
1. Download both python scripts
2. Go to UniProt/PDB to download the list of IDs you want to predict and save it as txt files where one line is one ID
3. (for UniProt) Download the fasta files to do the signal peptide removal. You might need to separate into several files since SignalP 6.0 can only batch predict 1000 sequences at a time and 5.0 predicts 5000 sequences at maximum
4. (for UniProt) Upload the fasta files to [SiganlP](https://services.healthtech.dtu.dk/services/SignalP-5.0/) (both 5.0 and 6.0 work) and download the output json files as {inputFilename}_1.json. If you have more than one json output file, download as {inputFilename}_1.json, {inputFilename}_2.json, {inputFilename}_3.json, etc.
5. Create a folder where the folder name is the same as the txt filename
6. Enter the following command into your command line
```
python3 slipp.py -i inputFilename.txt [-s numberofSignalPoutput] -o outputFilename.csv
```
-s numberofSignalPoutput: optional, an interger to specify how many json files you are ready to input for the signal peptide removal.

7. The 'predicition' column has the SLiPP-predicted probability of being lipid-binding proteins
