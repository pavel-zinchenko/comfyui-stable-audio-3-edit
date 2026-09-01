# ComfyUI Stable Audio 3 Editing

ComfyUI-native init-audio, inpainting, and continuation nodes for Stable Audio 3. The pack uses ComfyUI's existing checkpoint loader, Stable Audio text encoder, VAE, KSampler, model management, and audio decoder. It adds no dependencies and does not download models.

## Nodes

- **Stable Audio 3 Audio-to-Audio** prepares source audio and conditioning for variation or style transfer. Connect its latent to KSampler and use `denoise` to control the amount of change.
- **Stable Audio 3 Inpaint Region** selects one time range to regenerate.
- **Stable Audio 3 Inpaint Conditioning** accepts one or more region nodes, prepares the model's inpaint conditioning, and optionally preserves source latents outside the selected regions.

Source audio is resampled to 44.1 kHz, converted to stereo, and aligned for the Stable Audio 3 VAE. Audio-to-audio uses its explicit `seconds_total` to crop or extend the source. Inpainting automatically uses the greater of the source duration and all connected region end times.

## Recommended sampling

Use the settings from ComfyUI's Stable Audio 3 workflow:

- 8 steps
- CFG 1
- `lcm` sampler
- `simple` scheduler
- denoise 1.0 for inpainting
- denoise 0.8 as an audio-to-audio starting point

For continuation, create an inpaint region from the source duration to the target duration. The inpaint conditioning node extends the output duration to the region end automatically.

The nodes require a Stable Audio 3 VAE with 256 latent channels and 4096x temporal compression. They do not support the older Stable Audio 1 model.
