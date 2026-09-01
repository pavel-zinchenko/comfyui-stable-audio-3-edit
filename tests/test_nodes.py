import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

import torch


sys.path.insert(0, str(Path(__file__).parents[3]))


class _NodeOutput:
    def __init__(self, *args):
        self.args = args

    @property
    def result(self):
        return self.args


class _Autogrow:
    Type = dict


class _IO:
    ComfyNode = object
    NodeOutput = _NodeOutput
    Autogrow = _Autogrow


class _ComfyExtension:
    pass


comfy_api = types.ModuleType("comfy_api")
comfy_api_latest = types.ModuleType("comfy_api.latest")
comfy_api_latest.ComfyExtension = _ComfyExtension
comfy_api_latest.IO = _IO
sys.modules.setdefault("comfy_api", comfy_api)
sys.modules.setdefault("comfy_api.latest", comfy_api_latest)

module_path = Path(__file__).parents[1] / "nodes.py"
spec = importlib.util.spec_from_file_location("comfyui_stable_audio3_nodes", module_path)
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class FakeVAE:
    audio_sample_rate = 44100
    latent_channels = 256
    downscale_ratio = 4096

    def __init__(self):
        self.encoded = None

    def encode(self, audio):
        self.encoded = audio
        length = audio.shape[1] // self.downscale_ratio
        return torch.ones((audio.shape[0], self.latent_channels, length), dtype=audio.dtype, device=audio.device)


def conditioning():
    return [[torch.zeros((1, 1, 1)), {}]]


class StableAudio3NodeTests(unittest.TestCase):
    def test_inpaint_region_validation_returns_readable_error(self):
        self.assertTrue(nodes.StableAudio3InpaintRegion.validate_inputs(4.0, 8.0))
        self.assertEqual(
            nodes.StableAudio3InpaintRegion.validate_inputs(8.0, 4.0),
            "End time (4s) must be greater than start time (8s).",
        )
        self.assertEqual(
            nodes.StableAudio3InpaintRegion.validate_inputs(4.0, 4.0),
            "End time (4s) must be greater than start time (4s).",
        )

    def test_audio_to_audio_prepares_stereo_aligned_latent_and_duration_conditioning(self):
        vae = FakeVAE()
        audio = {"waveform": torch.ones((1, 1, 48000)), "sample_rate": 48000}

        positive, negative, latent = nodes.StableAudio3AudioToAudio.execute(
            conditioning(), conditioning(), vae, audio, 1.0
        ).result

        self.assertEqual(vae.encoded.shape, (1, 49152, 2))
        self.assertEqual(latent["samples"].shape, (1, 256, 12))
        self.assertEqual(latent["sample_rate"], 44100)
        self.assertEqual(positive[0][1]["seconds_total"], 1.0)
        self.assertEqual(negative[0][1]["seconds_total"], 1.0)

    def test_audio_to_audio_crops_long_audio_to_target_length(self):
        vae = FakeVAE()
        waveform = torch.arange(60000, dtype=torch.float32).reshape(1, 1, -1)
        audio = {"waveform": waveform, "sample_rate": 44100}

        nodes.StableAudio3AudioToAudio.execute(conditioning(), conditioning(), vae, audio, 1.0)

        self.assertEqual(vae.encoded.shape, (1, 49152, 2))
        self.assertTrue(torch.equal(vae.encoded[0, :, 0], waveform[0, 0, :49152]))
        self.assertTrue(torch.equal(vae.encoded[0, :, 0], vae.encoded[0, :, 1]))

    def test_inpaint_builds_model_conditioning_and_hard_preserve_mask(self):
        vae = FakeVAE()
        audio = {"waveform": torch.ones((1, 2, 441000)), "sample_rate": 44100}
        regions = {"region_0": (2.0, 4.0), "region_1": (6.0, 8.0)}

        positive, negative, latent, duration = nodes.StableAudio3InpaintConditioning.execute(
            conditioning(), conditioning(), vae, audio, regions, True
        ).result

        mask = positive[0][1]["concat_mask"]
        masked_latent = positive[0][1]["concat_latent_image"]
        self.assertEqual(mask.shape, (1, 1, 108))
        self.assertTrue(torch.equal(mask, negative[0][1]["concat_mask"]))
        self.assertEqual(mask[..., :22].sum(), 0)
        self.assertTrue(mask[..., 22:44].all())
        self.assertTrue(mask[..., 65:87].all())
        self.assertTrue(torch.equal(masked_latent, latent["samples"] * (1.0 - mask)))
        self.assertTrue(torch.equal(latent["noise_mask"], mask))
        self.assertEqual(positive[0][1]["seconds_total"], 10.0)
        self.assertEqual(duration, 10.0)

    def test_inpaint_model_native_mode_omits_noise_mask_and_merges_overlaps(self):
        vae = FakeVAE()
        audio = {"waveform": torch.ones((1, 2, 441000)), "sample_rate": 44100}
        regions = {"region_0": (2.0, 5.0), "region_1": (4.0, 6.0)}

        positive, _, latent, _ = nodes.StableAudio3InpaintConditioning.execute(
            conditioning(), conditioning(), vae, audio, regions, False
        ).result

        self.assertNotIn("noise_mask", latent)
        mask = positive[0][1]["concat_mask"]
        self.assertTrue(mask[..., 22:65].all())
        self.assertEqual(mask.sum(), 43)

    def test_inpaint_extends_duration_to_latest_region_end(self):
        vae = FakeVAE()
        audio = {"waveform": torch.ones((1, 2, 441000)), "sample_rate": 44100}

        positive, negative, latent, duration = nodes.StableAudio3InpaintConditioning.execute(
            conditioning(), conditioning(), vae, audio, {"region_0": (9.0, 11.0)}, True
        ).result

        self.assertEqual(positive[0][1]["seconds_total"], 11.0)
        self.assertEqual(negative[0][1]["seconds_total"], 11.0)
        self.assertEqual(latent["samples"].shape, (1, 256, 120))
        self.assertEqual(duration, 11.0)

    def test_inpaint_rejects_invalid_regions_and_duration_over_model_limit(self):
        vae = FakeVAE()
        audio = {"waveform": torch.ones((1, 2, 441000)), "sample_rate": 44100}
        cases = [((4.0, 4.0), "0 <= start < end"), ((4.0, 385.0), "384s maximum")]

        for region, error in cases:
            with self.subTest(region=region), self.assertRaisesRegex(ValueError, error):
                nodes.StableAudio3InpaintConditioning.execute(
                    conditioning(), conditioning(), vae, audio, {"region_0": region}, True
                )

    def test_example_workflows_use_nested_autogrow_socket_names(self):
        workflow_dir = Path(__file__).parents[1] / "example_workflows"

        for name in ("stable_audio_3_inpaint.json", "stable_audio_3_continuation.json"):
            with self.subTest(workflow=name):
                workflow = json.loads((workflow_dir / name).read_text(encoding="utf-8"))
                node = next(node for node in workflow["nodes"] if node["type"] == "StableAudio3InpaintConditioning")
                trim_node = next(node for node in workflow["nodes"] if node["type"] == "TrimAudioDuration")
                input_names = [input["name"] for input in node["inputs"]]
                region_link = next(link for link in workflow["links"] if link[0] == 8)
                duration_link = next(link for link in workflow["links"] if link[0] == 16)

                self.assertEqual(input_names[4:6], ["regions.region_0", "regions.region_1"])
                self.assertNotIn("region_0", input_names)
                self.assertNotIn("seconds_total", input_names)
                self.assertEqual(node["widgets_values"], [True])
                self.assertEqual(region_link[4], 4)
                self.assertEqual(node["outputs"][3]["name"], "duration")
                self.assertEqual(next(input for input in trim_node["inputs"] if input["name"] == "duration")["link"], 16)
                self.assertEqual(duration_link[1:5], [7, 3, 10, 2])

    def test_inpaint_requires_a_region(self):
        vae = FakeVAE()
        audio = {"waveform": torch.ones((1, 2, 44100)), "sample_rate": 44100}

        with self.assertRaisesRegex(ValueError, "at least one"):
            nodes.StableAudio3InpaintConditioning.execute(
                conditioning(), conditioning(), vae, audio, {}, True
            )

    def test_rejects_non_sa3_vae(self):
        vae = FakeVAE()
        vae.latent_channels = 64
        audio = {"waveform": torch.ones((1, 2, 44100)), "sample_rate": 44100}

        with self.assertRaisesRegex(ValueError, "Stable Audio 3 VAE"):
            nodes.StableAudio3AudioToAudio.execute(conditioning(), conditioning(), vae, audio, 1.0)


if __name__ == "__main__":
    unittest.main()
