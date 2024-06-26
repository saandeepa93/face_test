import sys
sys.path.append('.')

from imports import *
import torch 



def create_score_matrix(g_mode, probe_num, configs):
  template_save = f"./data/templates/{configs}"
  g1_rows = os.listdir(os.path.join(template_save, g_mode))
  g1_rows = [k for k in g1_rows if k.split('.')[-1]!="json"]

  probe_rows = os.listdir(os.path.join(template_save, probe_num))
  probe_rows = [k for k in probe_rows if k.split('.')[-1]!="json"]
  
  g1_sorted = sorted(g1_rows, key=lambda item: (int(item.split('_')[-1].split('.')[0])
                            , item))
  probe_sorted = sorted(probe_rows, key=lambda item: (int(item.split('_')[-1].split('.')[0])
                            , item))


  score_matrix = []
  mask_matrix = []
  meta_info = {}
  for idx1, g1_file in enumerate(tqdm(g1_sorted), 0):
    g_scores = []
    g_labels = []
    gallery_subject = g1_file.split('_')[0]

    g1_path = os.path.join(os.path.join(template_save, g_mode, g1_file))
    emb = torch.load(g1_path, map_location=torch.device('cuda'))
    
    for idx2, p_file in enumerate(probe_sorted, 0):
      p_path = os.path.join(os.path.join(template_save, probe_num, p_file))
      p_emb = torch.load(p_path, map_location=torch.device('cuda')).squeeze()

      probe_subject = p_file.split('_')[0] # self.df_probe.loc[row_idx]['subject_id']

      # v1 = emb / torch.sqrt(torch.sum(emb ** 2, -1, keepdims=True))
      # v2 = p_emb / torch.sqrt(torch.sum(p_emb ** 2, -1, keepdims=True))
      # similarity_score = torch.sum(v1 * v2, -1)

      similarity_score = (emb.contiguous().view(1, -1) @ p_emb.contiguous().view(1, -1).T).squeeze()

      label = int(gallery_subject == probe_subject)

      g_scores.append(similarity_score.cpu().item())
      g_labels.append(label)
    
    score_matrix.append(g_scores)
    mask_matrix.append(g_labels)
  
  score_matrix = np.array(score_matrix).T
  mask_matrix = np.array(mask_matrix).T

  root_dir = f"./data/score_files/{configs}"
  if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

  np.save(f"{root_dir}/face_score_{g_mode}_{pmode}.pkl", score_matrix)
  np.save(f"{root_dir}/face_mask_{g_mode}_{pmode}.pkl", mask_matrix)
    

if __name__ == "__main__":
  gmodes = ["Gallery1", "Gallery2"]
  # pmodes = ['face_incl_ctrl', 'face_incl_trt', 'face_rest_ctrl', 'face_rest_trt']
  pmodes = ['face_incl_ctrl', 'face_incl_trt']

  configs = "briar_9"
  for gmode in gmodes:
    for pmode in pmodes:
      create_score_matrix(gmode, pmode, configs)