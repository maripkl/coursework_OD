import os
import glob
import textgrid
from tqdm import tqdm

from pyannote.core import Annotation, Timeline, Segment
from pyannote.database.util import load_rttm


def texgrid2rttm(uri, textgrid_file, rttm_file):
    tg = textgrid.TextGrid.fromFile(textgrid_file)
    annotation = Annotation(uri) 
    spk = {}
    num_spk = 1
    cnt = 0
    for i in range(tg.__len__()):
        for j in range(tg[i].__len__()):
            if tg[i][j].mark:
                if tg[i].name not in spk:
                    spk[tg[i].name] = tg[i].name
                    num_spk += 1
                    
                starttime = tg[i][j].minTime
                endtime = tg[i][j].maxTime
                speaker = spk[tg[i].name]
                annotation[Segment(starttime, endtime), cnt] = speaker
                cnt += 1
                       
    with open(rttm_file, 'w') as f:
        annotation.write_rttm(f)

        
if __name__ == '__main__':
    
    ali_part = "Eval" # Eval, Test
    
    ali_part_root_dir = f"/home/alexey/Documents/datasets/AliMeeting/{ali_part}_Ali/{ali_part}_Ali_far"
    textgrid_dir = f"{ali_part_root_dir}/textgrid_dir"
    
    output_rttm_dir = f"./{ali_part}_Ali_far/rttm"
    os.makedirs(output_rttm_dir, exist_ok=True)


    for textgrid_path in tqdm(glob.glob(f"{textgrid_dir}/*")):
        basename = os.path.basename(textgrid_path)
        dirname = os.path.dirname(textgrid_path)
        uri = basename.split(".")[0]
        rttm_path = f"{output_rttm_dir}/{uri}.rttm"
        texgrid2rttm(uri, textgrid_path, rttm_path)



