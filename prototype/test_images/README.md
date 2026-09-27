# Test images for the Cross-Generator Test page

One folder per source. The folder named `real` holds real photos; **every other folder is treated as AI-generated**.

| Folder | What is inside | Where it came from |
|---|---|---|
| `real/` | 12 real photos | OpenFake test set (ImageNet / DOCCI photos) |
| `chatgpt_gpt-image-1.5/` | 12 images from GPT-Image-1.5, the model behind ChatGPT image generation | OpenFake test set |
| `gemini_nano-banana-pro/` | 4 images from Nano Banana Pro, Google's Gemini image model | OpenFake test set |
| `midjourney-v7/` | 8 Midjourney v7 images | OpenFake test set |
| `flux2/` | 8 FLUX.2 klein images | OpenFake test set |

**Add your own:** create new folders such as `team_chatgpt/`, `team_gemini/`, `team_real_phone/` and drop images in.
Real phone photos must go in a folder named `real` (or add them to the existing one).

Sample images: OpenFake (ComplexDataLab, CC BY-NC 4.0) — used here for non-commercial academic demonstration only.
