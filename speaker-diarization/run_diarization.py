import os
import sys
import glob2
from tqdm import tqdm
import pickle
import yaml
import time 
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


import torch
import torch.nn as nn
import torchaudio

from pyannote.core import Annotation, Timeline, Segment, SlidingWindow
from pyannote.database.util import load_rttm
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

sys.path.append("../voiceid")
sys.path.append("../StreamingSpeakerDiarization")

from data_io import load_audio
from backend import transform_embeddings, prepare_plda
from embeddings import prepare_model_brno, prepare_model_clova#, prepare_model_speechbrain
from segmentation import split_segments, split_overlap_part
from embeddings.extraction import extract_embeddings_wav
from sklearn.cluster import AgglomerativeClustering
from clustering import VB_diarization, VB_diarization_UP
import online_clustering
from links_cluster import LinksCluster

def diarization1(embedding='clova', clustering='online', dataset='aishell4', thr1=0.54, thr2=0.3, thr3=0.2, ps=1.0):
  np.random.seed(0)

  SAMPLE_RATE = 16000

  # config_common = yaml.load(open("config_common.yaml"), Loader=yaml.FullLoader)

  MODEL_PATH = "/content/voiceid/pretrained"

  model_path_brno = f"{MODEL_PATH}/brno/ResNet101_16kHz/nnet/raw_81.pth"
  model_path_clova = f"{MODEL_PATH}/clova/baseline_v2_ap.model"
  model_path_speechbrain = f"{MODEL_PATH}/speechbrain/embedding_model.ckpt"

  WIN_SIZE = 2.0
  STEP_SIZE = 1.0
  VAD_ORACLE = True

  DEVICE = "cuda:0"


  EMBEDDINGS_NAME = embedding # clova, speechbrain, brno
  CLUSTERING = clustering # ahc, links, online


  DATASET_NAME = dataset



  if DATASET_NAME == "aishell4": # https://www.openslr.org/111/

      data_root = '..' #config_common["datasets"]["AISHELL-4"]
      audio_dir = f"{data_root}/test/wav"
      rttm_dir = '/content/test/TextGrid'
      
      wav_list = glob2.glob(f"{audio_dir}/*.flac")
      uri2path = {os.path.splitext(os.path.basename(wav))[0]: wav for wav in wav_list}
      
      uri2ann_ref = {}
      for uri in uri2path:
          uri2ann_ref.update(load_rttm(f"{rttm_dir}/{uri}.rttm"))

  elif DATASET_NAME in ["AMI_test", "AMI_dev"]: # https://groups.inf.ed.ac.uk/ami/download/
      
      data_split = DATASET_NAME.split("_")[1]
      data_root = '..' #config_common["datasets"]["AMI"]
      rttm_dir = f"meta/AMI-diarization-setup/only_words/rttms/{data_split}"
      
      uri2ann_ref = {}
      rttm_list = glob2.glob(f"{rttm_dir}/*.rttm")
      for rttm_path in rttm_list:
          uri2ann_ref.update(load_rttm(rttm_path))
      
      uri2path = {}
      for uri in uri2ann_ref:
          uri2path[uri] = f"{data_root}/audio/{uri}.Mix-Headset.wav"
          
  elif DATASET_NAME in ["voxconverse_test", "voxconverse_dev"]: # https://github.com/joonson/voxconverse
      
      data_split = DATASET_NAME.split("_")[1]
      data_root = '.' #config_common["datasets"]["VoxConverse"]
      audio_dir = f"./audio"
      rttm_dir = f"meta/voxconverse/{data_split}"

      wav_list = glob2.glob(f"{audio_dir}/*.wav")
      uri2path = {os.path.splitext(os.path.basename(wav))[0]: wav for wav in wav_list}
      uri2ann_ref = {}
      for uri in uri2path:
          uri2ann_ref.update(load_rttm(f"{rttm_dir}/{uri}.rttm"))
      
  elif DATASET_NAME in ["Eval_ali_far", "Test_ali_far"]: # https://www.openslr.org/119/
      
      data_split = DATASET_NAME.split("_")[0]
      data_root = config_common["datasets"]["AliMeeting"]
      audio_dir = f"{data_root}/{data_split}_Ali/{data_split}_Ali_far/audio_dir"
      rttm_dir = f"meta/AliMeeting/{data_split}_Ali_far/rttm"
      
      wav_list = glob2.glob(f"{audio_dir}/*.wav")
      uri2path = {}
      for wav in wav_list:
          file_id = os.path.splitext(os.path.basename(wav))[0]
          parts = file_id.split("_")
          uri = "_".join(parts[:2])
          uri2path[uri] = wav
      
      uri2ann_ref = {}
      for uri in uri2path:
          uri2ann_ref.update(load_rttm(f"{rttm_dir}/{uri}.rttm"))
      
  else:
      print(f"Dataset '{DATASET_NAME}' not found")
      exit()


  if not VAD_ORACLE:
      
      # Parameters: https://huggingface.co/pyannote/segmentation#reproducible-research

      HYPER_PARAMETERS = {
        # onset/offset activation thresholds
        "onset": 0.5, "offset": 0.5,
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.1,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.1
      }
      
      vad_osd_joint = Model.from_pretrained(f"{MODEL_PATH}/frontend/pytorch_model.bin")

      vad_model = VoiceActivityDetection(segmentation=vad_osd_joint)
      vad_model.instantiate(HYPER_PARAMETERS)

      osd_model = OverlappedSpeechDetection(segmentation=vad_osd_joint)
      osd_model.instantiate(HYPER_PARAMETERS)
      

  # Example
  # ann_vad = vad_model("/home/alexey/Documents/datasets/AISHELL-4/test/wav/S_R003S04C01.flac") 
  # ann_osd = osd_model("/home/alexey/Documents/datasets/AISHELL-4/test/wav/S_R003S04C01.flac") 
  # ann_vad.extrude(ann_osd.get_timeline().support()).support()


  ### Voice activity and overlapped speech detection


  uri2vad = {}
  uri2osd = {}
  for uri, wav_path in uri2path.items():
      
      if VAD_ORACLE:
          vad = uri2ann_ref[uri].get_timeline().support()
          osd = vad.get_overlap()
      else:
          ann_vad = vad_model(uri2path[uri])
          vad = ann_vad.get_timeline().support()
          ann_osd = osd_model(uri2path[uri]) 
          osd = ann_osd.get_timeline().support()
          
      uri2vad[uri] = vad
      uri2osd[uri] = osd

  #TODO: cache annotations


  ### Embeddings extraction


  if EMBEDDINGS_NAME == "brno":
      emb_model = prepare_model_brno(model_path_brno, DEVICE, "onnx" if model_path_brno.endswith("onnx") else "pytorch")

  elif EMBEDDINGS_NAME == "clova":
      emb_model = prepare_model_clova(model_path_clova, DEVICE)

  elif EMBEDDINGS_NAME == "speechbrain":
      emb_model = prepare_model_speechbrain(model_path_speechbrain, DEVICE)


  cache_id = f"{DATASET_NAME}_emb-{EMBEDDINGS_NAME}_vad-{VAD_ORACLE}_win-{WIN_SIZE:.1f}_step-{STEP_SIZE:.1f}"
  cache_path = f"cache/{cache_id}.p"

  try:
      with open(cache_path, "rb") as f:
          uri2data = pickle.load(f)
  except:
      print(f"Not found in cache: '{cache_path}'")

      uri2data = {}
      for uri, wav_path in tqdm(list(uri2path.items())[:2]):

          vad_timeline = uri2vad[uri]
          osd_timeline = uri2osd[uri]
          vad_timeline = vad_timeline.extrude(osd_timeline).support() # (!!!) exclude segments with overlapped speech

          waveform, _ = load_audio(wav_path)

          segments = split_segments(vad_timeline, WIN_SIZE, STEP_SIZE)
          embeddings = extract_embeddings_wav(emb_model, waveform, DEVICE, segments, batch_size=160)

          uri2data[uri] = (embeddings, segments, waveform)
            
      os.makedirs("./cache", exist_ok=True)
      with open(cache_path, "wb") as f:
          pickle.dump(uri2data, f)
      

  ### Clustering


  uri2ann_hyp = {}

  for uri in tqdm(uri2data):
      start_time = time.time()

      
      embeddings_raw, segments, waveform = uri2data[uri]
      
      embeddings = transform_embeddings(embeddings_raw, EMBEDDINGS_NAME, f"{MODEL_PATH}/brno")


      if CLUSTERING == "ahc":
          
          #threshold = 0.1 # clova
          #threshold = -0.1 # speechbrain
          threshold = -0.2 # brno
          
          similarity_matrix = np.dot(embeddings, embeddings.T) - np.eye(embeddings.shape[0])
            
          distance_matrix = (-1.0) * similarity_matrix
          distance_threshold = (-1.0) * threshold
          cluster_model = AgglomerativeClustering(n_clusters=None,
                                                  distance_threshold=distance_threshold,
                                                  compute_full_tree=True,
                                                  affinity="precomputed",
                                                  linkage="complete")
          y_pred = cluster_model.fit_predict(distance_matrix)
          labels = cluster_model.labels_
      
      if CLUSTERING == 'links':
        links_cluster = LinksCluster(cluster_similarity_threshold=thr2, subcluster_similarity_threshold=thr3, pair_similarity_maximum=ps)
        labels = []
        for i in range(len(segments)):
          if time.time() - start_time > 20:
            return 1000, 1000, 'bad'
          labels += [links_cluster.predict(embeddings[i])]
        ann_hyp = Annotation(uri=uri)
        for segment, label in zip(segments, labels):
          ann_hyp[segment] = str(label)    
      if CLUSTERING == 'online':
        onlineCluster = online_clustering.OnlineClustering(uri, None, threshold=thr1)
        for i in range(len(segments)):
          # if np.mean(sad_score.crop(segment), axis=0)[1] < 0.5:
          #     continue
          if time.time() - start_time > 20:
            return 1000, 1000, 'bad'
          data = {}
          data['embedding'] = embeddings[i]
          data['segment'] = segments[i]
          data['indice'] = i
          onlineCluster.upadateCluster2(data)
        
        ann_hyp = onlineCluster.getAnnotations()
      if CLUSTERING =='streaming':
          from diart.blocks import OnlineSpeakerClustering, DelayedAggregation, Binarize
          from pyannote.core import notebook, Segment, SlidingWindow
          from pyannote.core import SlidingWindowFeature as SWF
          clustering = OnlineSpeakerClustering(0.5, 0.4, 1.5)
          #делаем слайдингвиндовфича из вэйфвормы? св(должно быть столько же сколько эмбеддингов)
          ans = []
          for i in range(len(segments)):
            cur_step = range(0,waveform.size(1),int(SAMPLE_RATE*STEP_SIZE))
            window = SlidingWindow(start=segments[i].start, duration = segments[i].end - segments[i].start, step=segments[i].end-segments[i].start)
            swf = SWF(waveform[:,cur_step[i]:min(waveform.size(1),cur_step[i]+int(WIN_SIZE*SAMPLE_RATE))].numpy().T, window)
            embed = torch.tensor(embeddings[i]).unsqueeze(0)
            ans += [clustering(swf, embed)] # ?
          # ann_hyp = Annotation(uri=uri)
          # for segment, label in zip(segments, labels):
          #     ann_hyp[segment] = str(label)    
      da = DelayedAggregation()
      ans = da(ans)
      bina = Binarize()
      ann_hyp = bina(ans)
      ann_hyp = split_overlap_part(ann_hyp.support()) # (!!!)
      uri2ann_hyp[uri] = ann_hyp


  ### Post-processing

  # Nothing for now, todo




  ### Performance metrics


  results_table = PrettyTable()
  results_table.title = DATASET_NAME
  results_table.field_names = ["Collar", "Skip overlap", "DER, %", "JER, %"]
  der = []
  jer = []
  for (collar, skip_overlap) in [(0, False), 
                                (0, True), 
                                (0.25, True)]:

      # Diarization Error Rate (DER)
      # NOTE: FA and Miss will be zero if skip_overlap=True and the Oracle VAD is used, that is, DER = Confusion
      der_metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
      for uri, ann_hyp in uri2ann_hyp.items():
          ann_ref = uri2ann_ref[uri]
          # print(ann_ref, ann_hyp)
          der_metric(ann_ref, ann_hyp)
          der.append(der_metric(ann_ref, ann_hyp))

      report = der_metric.report(display=False)
      DER = report["diarization error rate"]["%"]["TOTAL"]
    #   print(f"DER: {DER}%")
      # print('&&&&&&')
      # Jaccard Error Rate (JER)
      jer_metric = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
      for uri, ann_hyp in uri2ann_hyp.items():
          ann_ref = uri2ann_ref[uri]
          jer_metric(ann_ref, ann_hyp)
          jer.append(jer_metric(ann_ref, ann_hyp))

      report = jer_metric.report(display=False)
      JER = report["jaccard error rate"]["%"]["TOTAL"]
    #   print(f"JER: {JER}%")

      results_table.add_row([collar, skip_overlap, f"{DER:.2f}", f"{JER:.2f}"])
      
#   print(DER)
#   print(JER)
  # print(results_table)
  
  return DER, JER, 'ok'

