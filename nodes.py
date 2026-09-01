import math

import torch
import torchaudio

import node_helpers
from comfy_api.latest import ComfyExtension, IO
from typing_extensions import override


SA3_SAMPLE_RATE = 44100
SA3_LATENT_CHANNELS = 256
SA3_DOWNSCALE_RATIO = 4096
SA3_INPAINT_REGION = "SA3_INPAINT_REGION"
MAX_REGIONS = 16


def _validate_vae(vae):
    sample_rate = getattr(vae, "audio_sample_rate", None)
    latent_channels = getattr(vae, "latent_channels", None)
    downscale_ratio = getattr(vae, "downscale_ratio", None)
    if sample_rate != SA3_SAMPLE_RATE or latent_channels != SA3_LATENT_CHANNELS or downscale_ratio != SA3_DOWNSCALE_RATIO:
        raise ValueError("Stable Audio 3 nodes require the Stable Audio 3 VAE (44.1 kHz, 256 latent channels, 4096x compression).")


def _get_audio_waveform(audio):
    if audio is None:
        raise ValueError("Stable Audio 3: audio input is required.")

    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim != 3 or waveform.shape[1] < 1:
        raise ValueError("Stable Audio 3: audio waveform must have shape [batch, channels, samples].")
    if sample_rate <= 0:
        raise ValueError("Stable Audio 3: audio sample rate must be positive.")
    return waveform, sample_rate


def _prepare_audio_latent(vae, audio, seconds_total):
    _validate_vae(vae)
    waveform, sample_rate = _get_audio_waveform(audio)

    if sample_rate != SA3_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SA3_SAMPLE_RATE)

    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2]

    target_samples = round(seconds_total * SA3_SAMPLE_RATE)
    latent_length = math.ceil(target_samples / SA3_DOWNSCALE_RATIO)
    if latent_length % 2:
        latent_length += 1
    aligned_samples = latent_length * SA3_DOWNSCALE_RATIO

    prepared = waveform.new_zeros((waveform.shape[0], 2, aligned_samples))
    copy_length = min(waveform.shape[-1], aligned_samples)
    prepared[..., :copy_length] = waveform[..., :copy_length]
    latent = vae.encode(prepared.movedim(1, -1))
    return latent


def _set_seconds_total(positive, negative, seconds_total):
    values = {"seconds_total": seconds_total}
    return (
        node_helpers.conditioning_set_values(positive, values),
        node_helpers.conditioning_set_values(negative, values),
    )


def _audio_latent(samples):
    return {"samples": samples, "type": "audio", "sample_rate": SA3_SAMPLE_RATE}


class StableAudio3AudioToAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="StableAudio3AudioToAudio",
            display_name="Stable Audio 3 Audio-to-Audio",
            category="model/conditioning/stable audio 3",
            description="Encodes source audio for Stable Audio 3 variation and editing. Control the change strength with KSampler denoise.",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Vae.Input("vae"),
                IO.Audio.Input("audio"),
                IO.Float.Input("seconds_total", default=30.0, min=1.0, max=384.0, step=0.1),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, audio, seconds_total):
        latent = _prepare_audio_latent(vae, audio, seconds_total)
        positive, negative = _set_seconds_total(positive, negative, seconds_total)
        return IO.NodeOutput(positive, negative, _audio_latent(latent))


class StableAudio3InpaintRegion(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="StableAudio3InpaintRegion",
            display_name="Stable Audio 3 Inpaint Region",
            category="model/conditioning/stable audio 3",
            description="Selects a time range to regenerate with Stable Audio 3.",
            inputs=[
                IO.Float.Input("start_seconds", default=4.0, min=0.0, max=384.0, step=0.01),
                IO.Float.Input("end_seconds", default=8.0, min=0.0, max=384.0, step=0.01),
            ],
            outputs=[IO.Custom(SA3_INPAINT_REGION).Output(display_name="region")],
        )

    @classmethod
    def validate_inputs(cls, start_seconds, end_seconds):
        if start_seconds >= end_seconds:
            return f"End time ({end_seconds:g}s) must be greater than start time ({start_seconds:g}s)."
        return True

    @classmethod
    def execute(cls, start_seconds, end_seconds):
        if end_seconds <= start_seconds:
            raise ValueError("Stable Audio 3 inpaint region end must be after its start.")
        return IO.NodeOutput((float(start_seconds), float(end_seconds)))


class StableAudio3InpaintConditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="StableAudio3InpaintConditioning",
            display_name="Stable Audio 3 Inpaint Conditioning",
            category="model/conditioning/stable audio 3",
            description="Encodes source audio and conditions Stable Audio 3 to regenerate selected regions.",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Vae.Input("vae"),
                IO.Audio.Input("audio"),
                IO.Autogrow.Input(
                    "regions",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Custom(SA3_INPAINT_REGION).Input("region"),
                        prefix="region_",
                        min=1,
                        max=MAX_REGIONS,
                    ),
                    tooltip="Connect one or more Stable Audio 3 Inpaint Region nodes.",
                ),
                IO.Boolean.Input(
                    "preserve_outside_mask",
                    default=True,
                    advanced=True,
                    tooltip="Keep source latents fixed outside the selected regions. Disable for model-native inpainting that may alter surrounding audio.",
                ),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(display_name="latent"),
                IO.Float.Output(display_name="duration"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, audio, regions: IO.Autogrow.Type, preserve_outside_mask=True):
        region_values = [region for region in regions.values() if region is not None]
        if not region_values:
            raise ValueError("Stable Audio 3 inpainting requires at least one inpaint region.")

        for start_seconds, end_seconds in region_values:
            if start_seconds < 0 or end_seconds <= start_seconds:
                raise ValueError("Stable Audio 3 inpaint regions require 0 <= start < end.")

        waveform, sample_rate = _get_audio_waveform(audio)
        audio_duration = waveform.shape[-1] / sample_rate
        seconds_total = max(audio_duration, *(end_seconds for _, end_seconds in region_values))
        if seconds_total > 384.0:
            raise ValueError(f"Stable Audio 3 inpaint duration ({seconds_total:g}s) exceeds the 384s maximum.")

        latent = _prepare_audio_latent(vae, audio, seconds_total)
        generate_mask = latent.new_zeros((latent.shape[0], 1, latent.shape[-1]))
        for start_seconds, end_seconds in region_values:

            start_index = math.ceil(start_seconds * SA3_SAMPLE_RATE / SA3_DOWNSCALE_RATIO)
            end_index = math.ceil(end_seconds * SA3_SAMPLE_RATE / SA3_DOWNSCALE_RATIO)
            start_index = min(start_index, latent.shape[-1])
            end_index = min(end_index, latent.shape[-1])
            if start_index == end_index:
                raise ValueError("Stable Audio 3 inpaint region is shorter than one latent time step.")
            generate_mask[..., start_index:end_index] = 1.0

        masked_latent = latent * (1.0 - generate_mask)
        values = {
            "seconds_total": seconds_total,
            "concat_latent_image": masked_latent,
            "concat_mask": generate_mask,
        }
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)

        output_latent = _audio_latent(latent)
        if preserve_outside_mask:
            output_latent["noise_mask"] = generate_mask
        return IO.NodeOutput(positive, negative, output_latent, seconds_total)


class StableAudio3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            StableAudio3AudioToAudio,
            StableAudio3InpaintRegion,
            StableAudio3InpaintConditioning,
        ]


async def comfy_entrypoint() -> StableAudio3Extension:
    return StableAudio3Extension()
