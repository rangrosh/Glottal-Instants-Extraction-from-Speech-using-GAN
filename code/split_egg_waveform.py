import numpy as np 
import os 
from tqdm import tqdm 

def split_speech_signal(home_dir, desc, sub):

    dest_speech = 'Speech'
    dest_egg = 'EGG'

    if not os.path.exists(dest_speech):
        os.mkdir(dest_speech)
    if not os.path.exists(dest_egg):
        os.mkdir(dest_egg) 

    for i in tqdm(os.listdir(home_dir), desc = desc, ncols = 80):
        mod_i = sub + '_' + i 
        path_to_wav = os.path.join(home_dir, i)
        dest_speech_dir = os.path.join(dest_speech, mod_i)
        dest_egg_dir = os.path.join(dest_egg, mod_i) 

        os.system('ch_wave -c 0 -F 16000 {0} -o {1}'.format(path_to_wav, dest_speech_dir))
        os.system('ch_wave -c 1 -F 16000 {0} -o {1}'.format(path_to_wav, dest_egg_dir))
    

def main():

    # os.chdir('/home/harish/Documents/Python Scripts/SAP Project/Glottal-Instants-Extraction')
    home_dir = [
        'cmu_us_bdl_arctic-WAVEGG/cmu_us_bdl_arctic/orig/', 
        'cmu_us_jmk_arctic-WAVEGG/cmu_us_jmk_arctic/orig/', 
        'cmu_us_slt_arctic-WAVEGG/cmu_us_slt_arctic/orig/'
    ]

    desc = ["Extracting BDL", "Extracting JMK", "Extracting SLT"]

    subset = ['bdl', 'jmk', 'slt']

    for i in range(3):
        split_speech_signal(home_dir = home_dir[i], desc = desc[i], sub = subset[i])

if __name__ == "__main__":
    main()  