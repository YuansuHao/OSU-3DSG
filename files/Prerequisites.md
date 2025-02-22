### Prerequisites

Before running OSU-3DSG, make sure you have obtained the following checkpoints:

#### VLM Model
Use `git-lfs` to download weights of [LLaVA-Vicuna-7B Model](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b/tree/main) and [Qwen2.5-VL-72B-Instruct Model](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/tree/main)::
```bash
git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b
git clone https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct
```
#### CLIP Model
Use `git-lfs` to download weights of [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main):
```bash
git lfs install
git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```
#### MobileSAMv2
```bash 
gdown 1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE -O ~/weights/mobilesamv2/weight.zip
unzip ~/weights/mobilesamv2/weight.zip -d ~/weights/mobilesamv2/
cp ~/MobileSAM/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt ~/weights/mobilesamv2/weight/
```
<!-- Or download manually:
- Download weight.zip from [Google Drive](https://drive.usercontent.google.com/download?id=1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE&export=download&authuser=0).
- Extract the file and copy *Prompt_guided_Mask_Decoder.pt* to the weight folder. -->


The file structure looks like:
```
ckpt/
|–– llava-v1.6-vicuna-7b/
    |-- ...
|–– Qwen2.5-VL-72B-Instruct/
    |-- ...
|–– clip-vit-large-patch14-336/
    |-- dinov2_vits14_reg4_pretrain.pth
    |-- dinov2_vits14_reg4_linear_head.pth
|–– MobileSAMv2/weights/mobilesamv2/
    |-- Prompt_guided_Mask_Decoder.pt
```
