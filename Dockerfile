# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.5.0-base
# ------------------------------------------------------------------------------
# 1. INSTALL CUSTOM NODES
# ------------------------------------------------------------------------------
RUN comfy node install --exit-on-fail comfyui_controlnet_aux
RUN comfy node install --exit-on-fail ComfyUI_Ib_CustomNodes
RUN comfy node install --exit-on-fail comfyui_segment_anything
RUN comfy node install --exit-on-fail ComfyUI_tinyterraNodes@2.0.9
RUN comfy node install --exit-on-fail ComfyUI-Custom-Scripts
RUN comfy node install --exit-on-fail ComfyUI-DepthAnythingV2
RUN comfy node install --exit-on-fail ComfyUI-Easy-Use
RUN comfy node install --exit-on-fail ComfyUI-Frame-Interpolation
RUN comfy node install --exit-on-fail ComfyUI-GGUF@1.1.8
RUN comfy node install --exit-on-fail comfyui-int-and-float
RUN comfy node install --exit-on-fail ComfyUI-KJNodes
RUN comfy node install --exit-on-fail ComfyUI-Manager
RUN comfy node install --exit-on-fail Comfyui-Resolution-Master
RUN comfy node install --exit-on-fail ComfyUI-VideoHelperSuite
RUN comfy node install --exit-on-fail ComfyUI-WanVideoWrapper
RUN comfy node install --exit-on-fail facerestore_cf
RUN comfy node install --exit-on-fail RES4LYF
RUN comfy node install --exit-on-fail rgthree-comfy@1.0.2511270846
RUN comfy node install --exit-on-fail was-node-suite-comfyui@1.0.2
# ------------------------------------------------------------------------------
# 2. PREPARE FOR DRIVE DOWNLOADS
# ------------------------------------------------------------------------------
# Install gdown to handle Google Drive links reliably
RUN pip install gdown
# Create the loras directory explicitly to avoid path errors
RUN mkdir -p models/loras
# ------------------------------------------------------------------------------
# 3. DOWNLOAD MODELS FROM GOOGLE DRIVE (Casey & Instagirl)
# ------------------------------------------------------------------------------
# Casey.safetensors
# ID: 1yqAC9dOsEx-1_sRkBXz4VnI8B4oK2DAK
RUN gdown "https://drive.google.com/uc?id=1yqAC9dOsEx-1_sRkBXz4VnI8B4oK2DAK" -O models/loras/Casey.safetensors
# Instagirlv2.0_lownoise.safetensors
# ID: 14B_g3J2jNnnvCKoX8w6fx78SUAsaCU9g
RUN gdown "https://drive.google.com/uc?id=14B_g3J2jNnnvCKoX8w6fx78SUAsaCU9g" -O models/loras/Instagirlv2.0_lownoise.safetensors
# (Optional) Instagirl High - Commented out as it wasn't in the screenshot requirements
# RUN gdown "https://drive.google.com/uc?id=1bnuys1s5k3dQYgj9PzMDZu8NMgfPvG7X" -O models/loras/Instagirlv2.0_high.safetensors
# ------------------------------------------------------------------------------
# 4. DOWNLOAD MODELS FROM HUGGINGFACE
# ------------------------------------------------------------------------------
# Main Diffusion Model (Wan 2.2 GGUF)
RUN comfy model download \
    --url https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf \
    --relative-path models/diffusion_models \
    --filename Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf
# Text Encoder (T5)
RUN comfy model download \
    --url https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors \
    --relative-path models/clip \
    --filename umt5_xxl_fp16.safetensors
# VAE
RUN comfy model download \
    --url https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors \
    --relative-path models/vae \
    --filename wan_2.1_vae.safetensors
# Lightning LoRA
RUN comfy model download \
    --url https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors \
    --relative-path models/loras \
    --filename Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors
# ------------------------------------------------------------------------------
# 5. FINAL SETUP
# ------------------------------------------------------------------------------
