import numpy as np
import torch
import cv2
import os
import glob
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from peft import LoraConfig, get_peft_model

def yolo_to_mask(txt_path, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(txt_path): return mask
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) > 5:
                pts = np.array(parts[1:]).reshape(-1, 2)
                pts[:, 0] *= w; pts[:, 1] *= h
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask

def main():
    data_dir = "/content/drive/MyDrive/data_lake/03_gold/016_polyp_fast_diag_dataset/train"
    img_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.*")))
    label_paths = sorted(glob.glob(os.path.join(data_dir, "labels", "*.txt")))
    
    yolo_model = YOLO("/content/drive/MyDrive/models/trained/025_polyp_yolov12x_aug/train/weights/best.pt")
    
    sam2_checkpoint = "sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    
    # FREEZE all SAM2 base parameters
    for param in predictor.model.parameters():
        param.requires_grad = False
    
    # Configure LoRA for SAM2 Mask Decoder
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none"
    )
    predictor.model.sam_mask_decoder = get_peft_model(predictor.model.sam_mask_decoder, lora_config)
    print("Trainable parameters in Mask Decoder:")
    predictor.model.sam_mask_decoder.print_trainable_parameters()
    
    # Ensure gradients are enabled for the LoRA adapter
    predictor.model.sam_mask_decoder.train(True)
    
    optimizer = torch.optim.AdamW(params=predictor.model.sam_mask_decoder.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(img_paths, desc=f"Epoch {epoch+1}/{epochs}")
        for img_path in pbar:
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            gt_mask_np = yolo_to_mask(os.path.join(data_dir, "labels", Path(img_path).stem + ".txt"), h, w)
            if gt_mask_np.sum() == 0: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                results = yolo_model(img, conf=0.25, verbose=False)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
            
            if len(boxes) == 0: continue
            
            with torch.cuda.amp.autocast():
                # predictor.set_image modifies internal features; requires no_grad
                with torch.no_grad():
                    predictor.set_image(img_rgb)
                    box_tensor = torch.tensor(boxes, dtype=torch.float32, device="cuda")
                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=None, boxes=box_tensor, masks=None
                    )
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                    image_embeddings = predictor._features["image_embed"][-1].unsqueeze(0)
                    image_pe = predictor.model.sam_prompt_encoder.get_dense_pe()
                
                # Forward pass through LoRA-enabled mask decoder requires gradient tracking
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=True,
                    high_res_features=high_res_features,
                )
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                
                gt_mask = torch.tensor(gt_mask_np, dtype=torch.float32, device="cuda").unsqueeze(0)
                if prd_masks.shape[0] > 1:
                    prd_masks = prd_masks.max(dim=0, keepdim=True)[0]
                
                prd_prob = torch.sigmoid(prd_masks[:, 0])
                            bce = torch.nn.functional.binary_cross_entropy_with_logits(prd_masks[:, 0], gt_mask)
                            inter = (prd_prob * gt_mask).sum((1,2))
                            dice_loss = 1.0 - (2.*inter + 1e-5) / (prd_prob.sum((1,2)) + gt_mask.sum((1,2)) + 1e-5)
                            seg_loss = bce + dice_loss.mean()
                
                loss = seg_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
    os.makedirs("weights", exist_ok=True)
    # Save only the LoRA weights
    predictor.model.sam_mask_decoder.save_pretrained("weights/lora_finetuned")
    print("Training complete, LoRA adapter weights saved.")

if __name__ == "__main__":
    main()
