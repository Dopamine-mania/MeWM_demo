from tkinter import E
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import openai
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import base64
import argparse
import os
from pathlib import Path

from monai.networks.nets import DynUNet
from Segmentation.dynunet_pipeline.create_network import get_kernels_strides
from Segmentation.dynunet_pipeline.task_params import deep_supr_num
from dataset import ImageDataset
from torch.utils.data import DataLoader
import random
from tqdm import tqdm

from openai import OpenAI


class MockPolicy:
    def __init__(self, drug_set=None, embolism_set=None):
        self._drug_set = drug_set or ["Epirubicin", "Oxaliplatin", "Cisplatin"]
        self._embolism_set = embolism_set or ["Lipiodol", "Gelatin Sponge", "PVA"]

    def get_drug_embolism_sets(self, prompt):
        return list(self._drug_set), list(self._embolism_set)

    def get_drug_embolism_sets_with_image(self, prompt, image_bytes):
        return self.get_drug_embolism_sets(prompt)


def _normalize_actions(action, batch_size: int):
    if action is None:
        return [""] * batch_size
    if isinstance(action, str):
        return [action] * batch_size
    if isinstance(action, (list, tuple)):
        action = list(action)
        if len(action) == batch_size:
            return action
        if len(action) % batch_size == 0:
            step = len(action) // batch_size
            return [action[i * step] for i in range(batch_size)]
        return (action + [""] * batch_size)[:batch_size]
    return [str(action)] * batch_size


class MockSegmentationModel(nn.Module):
    def forward(self, x):
        # x: [B, 1, D, H, W] in [-1, 1] (after ScaleIntensityRanged)
        bg = torch.zeros_like(x)
        organ = (x > -0.7).float()
        tumor = (x > 0.4).float() * 2.0
        return torch.cat([bg, organ, tumor], dim=1)


class MockTumorGenerativeWorldModel(nn.Module):
    def __init__(self, noise_scale: float = 0.03):
        super().__init__()
        self.noise_scale = float(noise_scale)

    def forward(self, x, tumor_mask, action):
        # Make output depend on action but stay close to x.
        actions = _normalize_actions(action, x.shape[0])
        gen = x.clone()
        for i, a in enumerate(actions):
            h = (abs(hash(a)) % 1000) / 1000.0  # 0..1
            delta = (h - 0.5) * 2.0 * self.noise_scale
            gen[i] = gen[i] + delta * tumor_mask[i]
        blur_mask = tumor_mask.float()
        return gen, blur_mask


class MockInverseDynamics(nn.Module):
    def forward(self, x, gen_x, tumor_mask, pred_mask):
        # Risk = mean absolute change inside tumor + small penalty for predicted tumor volume.
        diff = (gen_x - x).abs()
        tumor_sum = tumor_mask.sum(dim=(1, 2, 3, 4)).clamp(min=1.0)
        diff_score = (diff * tumor_mask).sum(dim=(1, 2, 3, 4)) / tumor_sum
        pred_penalty = pred_mask.float().sum(dim=(1, 2, 3, 4)) * 1e-6
        return (diff_score + pred_penalty).view(-1)

class Policy:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def get_drug_embolism_sets(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        "You are a medical AI assistant. Based on clinical guidelines, generate a JSON with 'drug_set' and 'embolism_set' for TACE treatment.")},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            return result.get("drug_set", []), result.get("embolism_set", [])
        except Exception as e:
            print(f"[GPT-4o Text Prompt Error] {e}")
            return [], []

    def get_drug_embolism_sets_with_image(self, prompt, image_bytes):
        try:
            image_base64 = base64.b64encode(image_bytes.getvalue()).decode()
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        "You are a medical AI assistant. Based on the CT image and prompt, return a JSON with 'drug_set' and 'embolism_set' for TACE treatment.")},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]}
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            return result.get("drug_set", []), result.get("embolism_set", [])
        except Exception as e:
            print(f"[GPT-4o Image+Text Prompt Error] {e}")
            return [], []



class SegmentationModel(nn.Module):
    def __init__(self, model_path):
        super(SegmentationModel, self).__init__()
        task_id = 'custom'
        kernels, strides = get_kernels_strides(task_id)
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=False,
            deep_supr_num=deep_supr_num[task_id],
        )
        self.model.load_state_dict(torch.load(model_path)['state_dict'])

    def forward(self, x):
        seg_pred = self.model(x)
        return seg_pred


class TumorGenerativeWorldModel(nn.Module):
    def __init__(self, device, cfg):
        super(TumorGenerativeWorldModel, self).__init__()
        # Lazy import to keep mock mode lightweight.
        from Synthesis.Segmentation.TumorGeneration.utils import synt_model_prepare
        self.vqgan, self.tester = synt_model_prepare(
            device=device,
            cfg=cfg,
        )

    def forward(self, x, tumor_mask, action):
        from Synthesis.Segmentation.TumorGeneration.utils import synthesize_tumor
        generated_volumes, blur_mask = synthesize_tumor(
                ct_volume=x,
                tumor_mask=tumor_mask,
                # organ_type=cfg.model.organ,
                vqgan=self.vqgan,
                tester=self.tester,
                text_description=action
            )
        return generated_volumes, blur_mask


class InverseDynamics(nn.Module):
    def __init__(self, model_path):
        super(InverseDynamics, self).__init__()
        from Survival.config import create_arg_parser
        from Survival.model.aggregator_wMask import aggregator_wMask
        args = create_arg_parser()
        self.model = aggregator_wMask(args)
        loc = "cuda:0"
        self.model.load_state_dict(torch.load(model_path, map_location=loc)["state_dict"])

    def forward(self, x, gen_x, tumor_mask, pred_mask):
        risk_score,_,_,_ = self.model([x, gen_x], [tumor_mask, pred_mask])
        return risk_score


def get_conflicting_drugs(drug):
    # Rule 1
    conflicts = {
        'Oxaliplatin': ['Lobaplatin', 'Cisplatin'],
        'Lobaplatin': ['Oxaliplatin', 'Cisplatin'],
        'Cisplatin': ['Oxaliplatin', 'Lobaplatin'],
        'Epirubicin': ['Idarubicin', 'THP'],
        'Idarubicin': ['Epirubicin', 'THP'],
        'THP': ['Epirubicin', 'Idarubicin']
    }
    return conflicts.get(drug, [])

def get_conflicting_embolisms(emb):
    # Rule 2
    conflicts = {
        'Absolute Alcohol': ['PVA'],
        'PVA': ['Absolute Alcohol', 'Gelatin Sponge'],
        'Gelatin Sponge': ['PVA']
    }
    return conflicts.get(emb, [])

def has_conflicts(item, plan, is_drug=True):
    if is_drug:
        conflicts = get_conflicting_drugs(item)
    else:
        conflicts = get_conflicting_embolisms(item)
    
    return any(conflict in plan for conflict in conflicts)

def argmin_drug(fgm_model, hsurv_model, hseg_model, x, tumor_mask, drug_set, plan, scores, T):
    best_score = scores
    best_drug = None
    
    # 1. Filter out used drugs and conflicting drugs
    available_drugs = [d for d in drug_set if d not in plan and not has_conflicts(d, plan, is_drug=True)]
    if not available_drugs:
        return None, best_score
        
    # 2. Batch processing, each batch processes 2 drugs
    BATCH_SIZE = 1
    scaler = torch.cuda.amp.GradScaler()  
    
    for i in range(0, len(available_drugs), BATCH_SIZE):
        batch_drugs = available_drugs[i:i + BATCH_SIZE]
        
        # 3. Build the actions for the current batch
        base_action = ';'.join(plan)
        actions = []
        for d in batch_drugs:
            action = d if not base_action else base_action + ';' + d
            actions.extend([action] * T)
        
        # 4. Build the input for the current batch
        batch_size = len(batch_drugs)
        x_batch = x.repeat(batch_size, 1, 1, 1, 1)
        tumor_mask_batch = tumor_mask.repeat(batch_size, 1, 1, 1, 1)
        
        # 5. Use mixed precision to calculate the score of the current batch
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            gen_x, gen_mask = fgm_model(x_batch, tumor_mask_batch, actions)
            pred_mask = hseg_model(gen_x)
            pred_mask = torch.softmax(pred_mask, 1)
            pred_mask = torch.argmax(pred_mask, dim=1).unsqueeze(1)
            # print(gen_x.shape, pred_mask.shape, tumor_mask_batch.shape, tumor_mask_post_batch.shape)
            scores_batch = hsurv_model(x_batch, gen_x, tumor_mask_batch, pred_mask)
            scores_batch = scores_batch.view(batch_size, T).mean(dim=1)
        
        # 6. Update the best score
        min_score_idx = torch.argmin(scores_batch)
        min_score = scores_batch[min_score_idx]
        
        if min_score < best_score:
            best_score = min_score
            best_drug = batch_drugs[min_score_idx]
            print('score:', min_score, 'best_score:', best_score)
        
        # 7. Clean up memory
        del gen_x, gen_mask, pred_mask, scores_batch
        torch.cuda.empty_cache()
    
    return best_drug, best_score


def argmin_embolism(fgm_model, hsurv_model, hseg_model, x, tumor_mask, embolism_set, plan, scores, T):
    best_score = scores
    best_emb = None
    
    # 1. Filter out used embolisms and conflicting embolisms
    available_embs = [e for e in embolism_set if e not in plan and not has_conflicts(e, plan, is_drug=False)]
    if not available_embs:
        return None, best_score
    
    # 2. Batch processing, each batch processes 2 embolisms
    BATCH_SIZE = 1
    scaler = torch.cuda.amp.GradScaler()  
    
    for i in range(0, len(available_embs), BATCH_SIZE):
        batch_embs = available_embs[i:i + BATCH_SIZE]
        
        # 3. Build the actions for the current batch
        base_action = ';'.join(plan)
        actions = []
        for e in batch_embs:
            action = e + ';' if not base_action else base_action + ';' + e
            actions.extend([action] * T)
        
        # 4. Build the input for the current batch
        batch_size = len(batch_embs)
        x_batch = x.repeat(batch_size, 1, 1, 1, 1)
        tumor_mask_batch = tumor_mask.repeat(batch_size, 1, 1, 1, 1)
        
        # 5. Use mixed precision to calculate the score of the current batch
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            gen_x, gen_mask = fgm_model(x_batch, tumor_mask_batch, actions)
            pred_mask = hseg_model(gen_x)
            pred_mask = torch.softmax(pred_mask, 1)
            pred_mask = torch.argmax(pred_mask, dim=1).unsqueeze(1)
            scores_batch = hsurv_model(x_batch, gen_x, tumor_mask_batch, pred_mask)
            scores_batch = scores_batch.view(batch_size, T).mean(dim=1)
        
        # 6. Update the best score
        min_score_idx = torch.argmin(scores_batch)
        min_score = scores_batch[min_score_idx]
        
        if min_score < best_score:
            best_score = min_score
            best_emb = batch_embs[min_score_idx]
            print('score:', min_score, 'best_score:', best_score)
        
        # 7. Clean up memory
        del gen_x, gen_mask, pred_mask, scores_batch
        torch.cuda.empty_cache()
    
    return best_emb, best_score




def ct_tensor_to_png_bytes(ct_tensor):
    # ct_tensor: torch.Tensor, shape [1, D, H, W] or [D, H, W]
    if ct_tensor.ndim == 4:
        ct_tensor = ct_tensor.squeeze(0)
    D = ct_tensor.shape[0]
    mid_slice = ct_tensor[D // 2].cpu().numpy()
    img = (mid_slice + 1) / 2 * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf


##############################################################
# 3. Main function: TACE Protocol Exploration with MeWM 
##############################################################
def tace_protocol_exploration(
    x, tumor_mask,
    drug_set, embolism_set,
    fgm_model, hsurv_model, hseg_model,
    T, B, H_d, H_e
):
    """
    Implement the main inference logic.
    """
    # Initialize B plans, each starting with a random drug
    plans = [[] for _ in range(B)]
    scores = [float('inf') for _ in range(B)]

    # Drug horizon
    for h in range(H_d):
        for b in range(B):
            drug, score = argmin_drug(fgm_model, hsurv_model, hseg_model, x, tumor_mask, drug_set, plans[b], scores[b], T)
            if drug is not None:
                plans[b].append(drug)
                scores[b] = score
                
        min_score_idx = scores.index(min(scores))
        max_score_idx = scores.index(max(scores))
        plans[max_score_idx] = plans[min_score_idx].copy()
        scores[max_score_idx] = scores[min_score_idx]
    
    # cTACE  Rule 3
    for b in range(B):
        plans[b].append('Lipiodol')

    # Continue searching for additional embolisms
    for h in range(H_e-1):  
        for b in range(B):
            emb, score = argmin_embolism(fgm_model, hsurv_model, hseg_model, x, tumor_mask, embolism_set, plans[b], scores[b], T)
            if emb is not None:
                plans[b].append(emb)
                scores[b] = score
                
        min_score_idx = scores.index(min(scores))
        max_score_idx = scores.index(max(scores))
        plans[max_score_idx] = plans[min_score_idx].copy()
        scores[max_score_idx] = scores[min_score_idx]
    
    # Select the best plan
    min_score_idx = scores.index(min(scores))
    best_plan = plans[min_score_idx]
    best_score = scores[min_score_idx]
    
    return best_plan, best_score


##############################################################
# 4. Example inference script call
##############################################################
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "real"], default="mock", help="mock: 不依赖权重/联网，强行跑通链路；real: 按原始逻辑加载权重")
    ap.add_argument("--policy", choices=["mock", "openai"], default="mock", help="mock: 固定药物/栓塞集合；openai: 调用 OpenAI 生成集合")
    ap.add_argument("--openai_api_key", type=str, default=None, help="OpenAI Key（也可用环境变量 OPENAI_API_KEY）")
    ap.add_argument("--data_root", type=str, default="main/work/mock_data", help="数据根目录（用于拼接 TSV 相对路径）")
    ap.add_argument("--data_list_file", type=str, default="main/work/mock_data/lists/train_paired.txt", help="10 列 TSV 列表文件路径")
    ap.add_argument("--out_json", type=str, default="main/work/mock_outputs/best_plan.json", help="输出 JSON 路径")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--B", type=int, default=3)
    ap.add_argument("--H_d", type=int, default=1)
    ap.add_argument("--H_e", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode == "mock":
        fgm_model = MockTumorGenerativeWorldModel().to(device).eval()
        hseg_model = MockSegmentationModel().to(device).eval()
        hsurv_model = MockInverseDynamics().to(device).eval()
    else:
        fgm_model = TumorGenerativeWorldModel(device=str(device), cfg="Synthesis/model/ddpm.yaml").to(device).eval()
        hsurv_model = InverseDynamics(model_path="checkpoint_best.pth.tar").to(device).eval()
        hseg_model = SegmentationModel(model_path="Segmentation/runs/hcc.fold0.nnunet/model.pt").to(device).eval()

    prompt = (
        "You are a radiation oncologist, please **list potential TACE drug and embolism sets** based on the clinical guidelines and patient's pre-treatment CT image. Please follow the below description:"
        "###* * Task Description**"
        "1. Analyze the input CT images and output potential TACE chemotherapy drug and embolization material sets for treatment. You can include any drugs and embolisms that you think may be helpful for the treatment. Chemotherapy drugs and embolization materials are limited to those in the Action Base."
        "2. The TACE action set is output in JSON format, including treatment plan keywords such as drugs and embolization materials."
        "### **Action Base**"
        "#### **Chemotherapy Drugs**"
        "- Raltitrexed"
        "- Epirubicin"
        "- Oxaliplatin"
        "- Lobaplatin"
        "- Mitomycin"
        "- Idarubicin"
        "- Nedaplatin"
        "- Pirarubicin"
        "- Cisplatin"
        "- Idarubicin"
        "- THP"
        "#### ** Embolization Materials **"
        "- Lipiodol"
        "- Gelatin Sponge"
        "- PVA"
        "- Absolute Alcohol"
        "- NBCA"
        "---"
        "output format as JSON: {\"drug_set\": [...], \"embolism_set\": [...]}."
    )

    if args.policy == "openai":
        key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise SystemExit("[ERROR] policy=openai 需要提供 --openai_api_key 或设置环境变量 OPENAI_API_KEY")
        client = Policy(api_key=key)
    else:
        client = MockPolicy()
    
    # Data loading
    dataset = ImageDataset(args.data_root, args.data_list_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # Inference and save results
    results_dict = {}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, data_dict in tqdm(enumerate(dataloader)):
        x = data_dict['input_CT_pre'].to(device)
        tumor_mask = data_dict['label_CT_pre'].to(device)
        name = data_dict['name']
        if name[0] in results_dict:
            print(f"Processed {name}: {results_dict[name[0]]}")
            continue
        x = x.squeeze(0).unsqueeze(1)
        tumor_mask = tumor_mask.squeeze(0).unsqueeze(1)

        if args.policy == "openai":
            img_bytes = ct_tensor_to_png_bytes(x.detach().cpu().squeeze(1)[0])  # [B, 1, D, H, W] -> [D, H, W]
            drug_set, embolism_set = client.get_drug_embolism_sets_with_image(prompt, img_bytes)
        else:
            drug_set, embolism_set = client.get_drug_embolism_sets(prompt)
        print("Policy generated drug_set:", drug_set)
        print("Policy generated embolism_set:", embolism_set)

        T = int(x.shape[0])
        with torch.no_grad():
            best_plan, best_risk = tace_protocol_exploration(
                x, tumor_mask,
                drug_set, embolism_set,
                fgm_model, hsurv_model, hseg_model,
                T, args.B, args.H_d, args.H_e
            )
        
        # Convert tensor to float
        best_risk = best_risk.item() if isinstance(best_risk, torch.Tensor) else float(best_risk)
        
        # Add new results
        results_dict[name[0]] = {
            'keywords': best_plan,
            'best_risk': best_risk
        }
        out_path.write_text(json.dumps(results_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Processed {name}: {results_dict[name[0]]}")
