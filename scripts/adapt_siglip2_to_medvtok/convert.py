#!/usr/bin/env python3
"""Lossless SigLIP2 NaFlex -> MedVTok encoder mapping, with forward verification.

No files in models/ are modified. See README.md for the equivalence contract,
the required runtime settings, temporal initialization, and pooling limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE = Path(
    "/inspire/qb-ilm/project/qproject-multimedicine/public/clma/weights/"
    "siglip2-so400m-patch16-naflex"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "converted"
ENCODER_FILE = "encoder.safetensors"
POOLING_FILE = "pooling_head.safetensors"
MANIFEST_FILE = "manifest.json"


def architecture_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in ("models/medvtok.py", "models/transformer.py", "models/utils.py")
    }


def read_vision_config(source: Path):
    from transformers import Siglip2VisionConfig

    config = Siglip2VisionConfig.from_pretrained(source, local_files_only=True)
    if config.hidden_act != "gelu_pytorch_tanh":
        raise ValueError(f"MedVTok uses tanh GELU; got {config.hidden_act!r}")
    if config.attention_dropout != 0.0:
        raise ValueError("Nonzero attention dropout is not equivalent during training")
    if math.isqrt(config.num_patches) ** 2 != config.num_patches:
        raise ValueError("The position embedding must have a square base grid")
    return config


def read_source_vision(source: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Only memory-map vision tensors; never instantiate the large text tower."""
    state = {}
    ignored = []
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("vision_model."):
                state[key.removeprefix("vision_model.")] = f.get_tensor(key)
            elif key.startswith("text_model.") or key in ("logit_scale", "logit_bias"):
                ignored.append(key)
            else:
                raise ValueError(f"Unrecognized source tensor: {key}")
    if not state:
        raise ValueError("No vision_model.* tensors found")
    return state, ignored


def map_vision_weights(
    vision: Mapping[str, torch.Tensor], config, temporal_patch_size: int, temporal_init: str
) -> tuple[dict, dict, list]:
    if temporal_patch_size < 1:
        raise ValueError("temporal_patch_size must be positive")
    if temporal_init not in ("first", "average"):
        raise ValueError("temporal_init must be 'first' or 'average'")
    encoder, pooling, mapping, consumed = {}, {}, [], set()

    def put(destination, source_keys, operation="copy"):
        source_keys = [source_keys] if isinstance(source_keys, str) else source_keys
        for key in source_keys:
            if key not in vision or key in consumed:
                raise ValueError(f"Missing or multiply consumed vision weight: {key}")
        tensors = [vision[key] for key in source_keys]
        if operation == "concat_qkv":
            value = torch.cat(tensors, dim=0)
        elif operation == "temporal_first":
            original = tensors[0]
            value = original.new_zeros(original.shape[0], original.shape[1] * temporal_patch_size)
            value[:, : original.shape[1]] = original
        elif operation == "temporal_average":
            value = tensors[0].repeat(1, temporal_patch_size) / temporal_patch_size
        else:
            value = tensors[0]
        encoder[destination] = value.contiguous()
        consumed.update(source_keys)
        mapping.append({"sources": source_keys, "destination": destination, "operation": operation})

    put("input_layer.weight", "embeddings.patch_embedding.weight", f"temporal_{temporal_init}")
    put("input_layer.bias", "embeddings.patch_embedding.bias")
    put("pos_embedder.position_embedding.weight", "embeddings.position_embedding.weight")
    for i in range(config.num_hidden_layers):
        src, dst = f"encoder.layers.{i}", f"blocks.{i}"
        for suffix in ("weight", "bias"):
            for n in (1, 2):
                put(f"{dst}.norm{n}.{suffix}", f"{src}.layer_norm{n}.{suffix}")
            put(
                f"{dst}.attn.to_qkv.{suffix}",
                [f"{src}.self_attn.{projection}_proj.{suffix}" for projection in ("q", "k", "v")],
                "concat_qkv",
            )
            put(f"{dst}.attn.to_out.{suffix}", f"{src}.self_attn.out_proj.{suffix}")
            put(f"{dst}.mlp.mlp.0.{suffix}", f"{src}.mlp.fc1.{suffix}")
            put(f"{dst}.mlp.mlp.2.{suffix}", f"{src}.mlp.fc2.{suffix}")
    for suffix in ("weight", "bias"):
        put(f"post_layernorm.{suffix}", f"post_layernorm.{suffix}")
    for key, tensor in vision.items():
        if key.startswith("head."):
            pooling[key.removeprefix("head.")] = tensor.contiguous()
            consumed.add(key)
            mapping.append({"sources": [key], "destination": key.removeprefix("head."),
                            "operation": "copy", "file": POOLING_FILE})
    if set(vision) != consumed:
        raise ValueError(f"Unmapped vision weights: {sorted(set(vision) - consumed)}")
    if not pooling:
        raise ValueError("Expected SigLIP2 attention pooling head weights")
    return encoder, pooling, mapping


def encoder_kwargs(config, temporal_patch_size: int) -> dict:
    return {
        "in_channels": config.num_channels * config.patch_size**2 * temporal_patch_size,
        "model_channels": config.hidden_size,
        "num_blocks": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "mlp_channels": config.intermediate_size,
        "attn_mode": "full",
        "qk_rms_norm": False,
        "use_bias": True,
        "use_rms_norm": False,
    }


def check_encoder_state(state: Mapping, kwargs: dict) -> None:
    from models.medvtok import MedVTokEncoder

    # Shape/key validation without initializing another 400M parameter model.
    with torch.device("meta"):
        template = MedVTokEncoder(**kwargs)
    expected = template.state_dict()
    if set(expected) != set(state):
        raise ValueError(f"Encoder key mismatch: missing={sorted(set(expected) - set(state))}, "
                         f"unexpected={sorted(set(state) - set(expected))}")
    for key, tensor in state.items():
        if tensor.shape != expected[key].shape or tensor.dtype != torch.float32:
            raise ValueError(f"Unexpected shape/dtype for {key}: {tensor.shape}, {tensor.dtype}; "
                             f"expected {expected[key].shape}, float32")


def convert(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, temporal_patch_size=4, temporal_init="first") -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    hashes = architecture_hashes()
    config = read_vision_config(source)
    vision, ignored = read_source_vision(source)
    encoder, pooling, mapping = map_vision_weights(vision, config, temporal_patch_size, temporal_init)
    kwargs = encoder_kwargs(config, temporal_patch_size)
    check_encoder_state(encoder, kwargs)
    # Validate the complete pooling state too, rather than silently preserving unknown keys.
    from transformers.models.siglip2.modeling_siglip2 import Siglip2MultiheadAttentionPoolingHead

    with torch.device("meta"):
        head = Siglip2MultiheadAttentionPoolingHead(config)
    head.load_state_dict(pooling, strict=True, assign=True)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in (ENCODER_FILE, POOLING_FILE, MANIFEST_FILE)]
    # A report from an earlier conversion does not verify the new bundle.
    (output / "verification.json").unlink(missing_ok=True)
    metadata = {"source": str(source), "format": "siglip2-to-medvtok-v1"}
    save_file(encoder, paths[0], metadata=metadata)
    save_file(pooling, paths[1], metadata=metadata)
    manifest = {
        "format": metadata["format"], "source": str(source),
        "vision_config": config.to_dict(), "encoder_kwargs": kwargs,
        "runtime": {"layer_norm_eps": config.layer_norm_eps, "use_rope": False,
                    "qk_rms_norm": False, "activation": "gelu_pytorch_tanh"},
        "input": {"patch_size": [temporal_patch_size, config.patch_size, config.patch_size],
                  "flatten_order": "pt ph pw c", "temporal_init": temporal_init,
                  "hf_bridge_temporal_fill": "zero" if temporal_init == "first" else "repeat"},
        "counts": {"source_vision_tensors": len(vision), "encoder_tensors": len(encoder),
                   "pooling_tensors": len(pooling), "ignored_nonvision_tensors": len(ignored)},
        "ignored_nonvision_keys": ignored, "mapping": mapping,
        "architecture_sha256": hashes,
    }
    paths[2].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if architecture_hashes() != hashes:
        raise RuntimeError("MedVTok source files changed during conversion")
    print(f"Converted all {len(vision)} vision tensors -> {output}", flush=True)
    return manifest


def read_manifest(bundle) -> dict:
    manifest = json.loads((Path(bundle) / MANIFEST_FILE).read_text())
    if manifest.get("format") != "siglip2-to-medvtok-v1":
        raise ValueError("Unrecognized bundle format")
    return manifest


def configure_encoder_for_siglip2(encoder, manifest: dict, *, enable_rope=False) -> None:
    """Set runtime attributes only; never replace forward methods or architecture modules.

    These attributes are not in a state_dict; call this again after constructing a
    model and reloading any training checkpoint. Enabling RoPE ends the equivalence
    guarantee and is solely an initialization option for later MedVTok training.
    """
    from models.medvtok import MedVTokEncoder
    from models.transformer import LayerNorm32

    if not isinstance(encoder, MedVTokEncoder):
        raise TypeError("Pass a MedVTokEncoder (e.g. model.encoder)")
    kwargs = manifest["encoder_kwargs"]
    if (encoder.input_layer.in_features != kwargs["in_channels"]
            or encoder.input_layer.out_features != kwargs["model_channels"]
            or len(encoder.blocks) != kwargs["num_blocks"]):
        raise ValueError(f"Incompatible encoder; required constructor arguments: {kwargs}")
    norms = [encoder.post_layernorm]
    for block in encoder.blocks:
        attn = block.attn
        if (attn.num_heads != kwargs["num_heads"] or attn.qk_rms_norm
                or attn.attn_mode != "full" or attn._type != "self"
                or block.mlp_channels != kwargs["mlp_channels"]):
            raise ValueError("Requires full self-attention, correct head/MLP dimensions, and no QK RMSNorm")
        activation = block.mlp.mlp[1]
        if not isinstance(activation, torch.nn.GELU) or activation.approximate != "tanh":
            raise ValueError("Requires GELU(approximate='tanh')")
        if enable_rope and not hasattr(attn, "rope"):
            raise ValueError("The encoder was constructed without RoPE")
        norms.extend((block.norm1, block.norm2))
    if not all(isinstance(norm, LayerNorm32) for norm in norms):
        raise ValueError("Requires MedVTok LayerNorm32; RMSNorm cannot represent SigLIP2 LayerNorm")
    if encoder.pos_embedder is None:
        raise ValueError("Requires learned absolute position embeddings")
    for norm in norms:
        norm.eps = manifest["runtime"]["layer_norm_eps"]
    for block in encoder.blocks:
        block.attn.use_rope = bool(enable_rope)


def load_into_encoder(encoder, bundle=DEFAULT_OUTPUT, *, enable_rope=False) -> dict:
    """Strictly load an existing encoder and apply indispensable runtime settings."""
    manifest = read_manifest(bundle)
    state = load_file(str(Path(bundle) / ENCODER_FILE), device="cpu")
    check_encoder_state(state, manifest["encoder_kwargs"])
    configure_encoder_for_siglip2(encoder, manifest, enable_rope=enable_rope)
    encoder.load_state_dict(state, strict=True)
    return manifest


def build_encoder(bundle=DEFAULT_OUTPUT, *, device="cpu", enable_rope=False):
    """Return (MedVTokEncoder, manifest), with exact forward settings by default."""
    from models.medvtok import MedVTokEncoder

    manifest = read_manifest(bundle)
    encoder = MedVTokEncoder(**manifest["encoder_kwargs"])
    load_into_encoder(encoder, bundle, enable_rope=enable_rope)
    return encoder.to(device), manifest


def load_pooling_head(bundle=DEFAULT_OUTPUT, *, device="cpu"):
    """The original MAP head, kept outside MedVTok's incompatible mean pool head."""
    from transformers import Siglip2VisionConfig
    from transformers.models.siglip2.modeling_siglip2 import Siglip2MultiheadAttentionPoolingHead

    manifest = read_manifest(bundle)
    config = Siglip2VisionConfig(**manifest["vision_config"])
    config._attn_implementation = "sdpa"
    head = Siglip2MultiheadAttentionPoolingHead(config)
    head.load_state_dict(load_file(str(Path(bundle) / POOLING_FILE)), strict=True)
    return head.to(device)


def pool_features(head, features: torch.Tensor, seqlens: list[int]) -> torch.Tensor:
    if not seqlens or min(seqlens) <= 0 or sum(seqlens) != features.shape[0]:
        raise ValueError("Expected positive seqlens covering every feature")
    padded = features.new_zeros(len(seqlens), max(seqlens), features.shape[-1])
    mask = torch.zeros(padded.shape[:2], dtype=torch.bool, device=features.device)
    offset = 0
    for i, length in enumerate(seqlens):
        padded[i, :length] = features[offset:offset + length]
        mask[i, :length] = True
        offset += length
    return head(padded, attention_mask=mask)


def siglip2_inputs_to_medvtok(inputs: Mapping, manifest: dict):
    """Bridge actual HF processor outputs, preserving resize/normalization exactly.

    NaFlex produces row-major full rectangular grids followed by padding. Drop
    that padding before MedVTok attention. The four coordinates are t, h, w, z.
    """
    pixels, mask, shapes = (inputs[key] for key in ("pixel_values", "pixel_attention_mask", "spatial_shapes"))
    patch_t, patch_h, patch_w = manifest["input"]["patch_size"]
    channels = manifest["vision_config"]["num_channels"]
    if (pixels.ndim != 3 or pixels.shape[-1] != patch_h * patch_w * channels
            or mask.shape != pixels.shape[:2] or shapes.shape != (pixels.shape[0], 2)):
        raise ValueError("Invalid NaFlex processor tensor shapes")
    features, coordinates, lengths = [], [], []
    for i, (height, width) in enumerate(shapes.tolist()):
        length = height * width
        if height <= 0 or width <= 0 or length > pixels.shape[1]:
            raise ValueError("Invalid spatial shape")
        if not bool(mask[i, :length].all()) or bool(mask[i, length:].any()):
            raise ValueError("Expected a complete rectangular grid and suffix-only padding")
        valid = pixels[i, :length]
        if manifest["input"]["hf_bridge_temporal_fill"] == "repeat":
            expanded = valid.unsqueeze(1).expand(-1, patch_t, -1).reshape(length, -1)
        else:
            expanded = valid.new_zeros(length, patch_t, valid.shape[-1])
            expanded[:, 0] = valid
            expanded = expanded.reshape(length, -1)
        indices = torch.arange(length, device=pixels.device)
        zeros = torch.zeros_like(indices)
        coords = torch.stack((torch.full_like(indices, i), zeros,
                              indices // width, indices % width, zeros), dim=-1)
        features.append(expanded)
        coordinates.append(coords)
        lengths.append(length)
    if not lengths:
        raise ValueError("Empty image batch")
    return torch.cat(features), torch.cat(coordinates), lengths


def _unpad(tensor: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    return torch.cat([tensor[i, :length] for i, length in enumerate(lengths)])


def _compare(actual, expected, *, atol, rtol) -> dict:
    actual, expected = actual.detach().float(), expected.detach().float()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    difference = actual - expected
    largest = difference.abs().reshape(-1).argmax()
    reference_at_largest = expected.reshape(-1)[largest].abs()
    spacing = torch.nextafter(reference_at_largest, reference_at_largest.new_tensor(float("inf"))) - reference_at_largest
    return {"max_abs": difference.abs().max().item(),
            "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-12)).item(),
            "max_reference_abs": expected.abs().max().item(),
            "reference_abs_at_max_error": reference_at_largest.item(),
            "ulp_at_max_abs_error": (difference.abs().max() / spacing.clamp_min(1e-38)).item(),
            "max_tolerance_fraction": (difference.abs() / (atol + rtol * expected.abs())).max().item()}


@torch.inference_mode()
def verify(bundle=DEFAULT_OUTPUT, *, source=None, device="cpu", shapes=None, atol=5e-4, rtol=3e-4) -> dict:
    """Compare every saved weight and every actual block against Hugging Face."""
    from transformers import Siglip2VisionModel
    from models.transformer import get_attn_backend, set_attn_backend

    bundle = Path(bundle)
    manifest = read_manifest(bundle)
    source = Path(source or manifest["source"])
    hashes = architecture_hashes()
    config = read_vision_config(source)
    expected_kwargs = encoder_kwargs(config, manifest["input"]["patch_size"][0])
    if manifest["encoder_kwargs"] != expected_kwargs or manifest["runtime"]["layer_norm_eps"] != config.layer_norm_eps:
        raise ValueError("Source configuration differs from the conversion manifest")
    vision, ignored = read_source_vision(source)
    expected, expected_pool, mapping = map_vision_weights(
        vision, config, manifest["input"]["patch_size"][0], manifest["input"]["temporal_init"]
    )
    if mapping != manifest["mapping"] or len(ignored) != manifest["counts"]["ignored_nonvision_tensors"]:
        raise ValueError("Source coverage differs from the conversion manifest")
    covered_sources = {key for entry in mapping for key in entry["sources"]}
    if covered_sources != set(vision):
        raise AssertionError("Verification must cover every source vision tensor")
    checked_weights = 0
    for filename, expected_state in ((ENCODER_FILE, expected), (POOLING_FILE, expected_pool)):
        saved = load_file(str(bundle / filename))
        if set(saved) != set(expected_state):
            raise AssertionError(f"Wrong saved keys in {filename}")
        for key, value in saved.items():
            torch.testing.assert_close(value, expected_state[key], atol=0, rtol=0, check_dtype=True)
            checked_weights += 1
        del saved
    print(f"Verified {checked_weights} saved tensors, covering all {len(vision)} source vision tensors", flush=True)
    del expected, expected_pool
    config._attn_implementation = "sdpa"
    reference = Siglip2VisionModel(config).eval()
    reference.vision_model.load_state_dict(vision, strict=True)
    reference = reference.to(device)
    encoder, _ = build_encoder(bundle, device=device)
    encoder.eval()
    head = load_pooling_head(bundle, device=device).eval()
    del vision
    shapes = shapes or [(16, 16), (12, 20), (20, 12), (24, 24), (1, 1)]
    lengths = [height * width for height, width in shapes]
    generator = torch.Generator(device="cpu").manual_seed(20261007)
    pixels = torch.rand(len(shapes), max(lengths), config.num_channels * config.patch_size**2,
                        generator=generator) * 2 - 1
    mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    # Deliberately large padding values make accidental padding attention visible.
    pixels[~mask] = 19.0
    inputs = {"pixel_values": pixels.to(device), "pixel_attention_mask": mask.to(device),
              "spatial_shapes": torch.tensor(shapes, device=device)}
    feats, coords, seqlens = siglip2_inputs_to_medvtok(inputs, manifest)
    stages, reference_inputs, handles = {}, {}, []

    def source_hook(name):
        def hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            stages[name] = _unpad(tensor, lengths).detach()
            if name.startswith("block_"):
                reference_inputs[name] = _unpad(args[0], lengths).detach()
        return hook

    results, independent_results = {}, {}
    comparison_pass = "cumulative"
    block_atol, block_rtol = 3e-5, 3e-5

    def target_hook(name, pre=False):
        def hook(module, args, output=None):
            if pre and comparison_pass == "independent":
                return
            tensor = args[0] if pre else (output[0] if isinstance(output, tuple) else output)
            destination = independent_results if comparison_pass == "independent" else results
            abs_tolerance, rel_tolerance = (block_atol, block_rtol) if comparison_pass == "independent" else (atol, rtol)
            try:
                destination[name] = _compare(tensor, stages[name], atol=abs_tolerance, rtol=rel_tolerance)
            except AssertionError as error:
                raise AssertionError(f"{comparison_pass} forward mismatch at {name}: {error}") from error
            print(f"{comparison_pass} {name}: max_abs={destination[name]['max_abs']:.6g}", flush=True)
        return hook

    handles.append(reference.vision_model.embeddings.register_forward_hook(source_hook("embeddings")))
    handles.append(encoder.blocks[0].register_forward_pre_hook(target_hook("embeddings", pre=True)))
    for i, (src_block, dst_block) in enumerate(zip(reference.vision_model.encoder.layers, encoder.blocks)):
        name = f"block_{i:02d}"
        handles.append(src_block.register_forward_hook(source_hook(name)))
        handles.append(dst_block.register_forward_hook(target_hook(name)))
    handles.append(reference.vision_model.post_layernorm.register_forward_hook(source_hook("post_layernorm")))
    handles.append(encoder.post_layernorm.register_forward_hook(target_hook("post_layernorm")))
    previous_backend = get_attn_backend()
    set_attn_backend("pytorch")
    try:
        reference_output = reference(**inputs)
        # Isolate each block from preceding rounding errors, using exactly the
        # reference block's input, with a tighter tolerance than the full stack.
        comparison_pass = "independent"
        for i, block in enumerate(encoder.blocks):
            block(reference_inputs[f"block_{i:02d}"], coords, seqlens)
        comparison_pass = "cumulative"
        actual, out_coords, out_seqlens = encoder(feats, coords, seqlens)
        if (not torch.equal(coords, out_coords) or seqlens != out_seqlens
                or set(results) != set(stages) or len(independent_results) != len(encoder.blocks)):
            raise AssertionError("Encoder changed coordinates/seqlens or missed verification stages")
        results["last_hidden_state"] = _compare(
            actual, _unpad(reference_output.last_hidden_state, lengths), atol=atol, rtol=rtol
        )
        results["pooler_output"] = _compare(
            pool_features(head, actual, seqlens), reference_output.pooler_output, atol=atol, rtol=rtol
        )
    finally:
        for handle in handles:
            handle.remove()
        set_attn_backend(previous_backend)
    if architecture_hashes() != hashes:
        raise RuntimeError("MedVTok source files changed during verification")
    report = {"passed": True, "device": str(device), "dtype": "float32",
              "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
              "shapes": shapes, "valid_tokens": sum(lengths), "padded_tokens": len(shapes) * max(lengths) - sum(lengths),
              "atol": atol, "rtol": rtol, "independent_block_atol": block_atol,
              "independent_block_rtol": block_rtol, "all_source_vision_tensors_covered": True,
              "source_vision_tensors": manifest["counts"]["source_vision_tensors"],
              "saved_tensors_checked_exactly": checked_weights,
              "architecture_sources_unchanged": True, "architecture_sha256": hashes,
              "stages": results, "independent_blocks": independent_results}
    (bundle / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(encoder.blocks)} blocks, patch features, and original MAP pooling; "
          f"final max_abs={results['last_hidden_state']['max_abs']:.6g}; "
          f"pooling max_abs={results['pooler_output']['max_abs']:.6g}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert_parser = subparsers.add_parser("convert", help="Convert every vision tensor, excluding the text tower/logits")
    convert_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    convert_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    convert_parser.add_argument("--temporal-patch-size", type=int, default=4,
                                help="1: exact 2D; 4: existing MedVTok input width (default)")
    convert_parser.add_argument("--temporal-init", choices=("first", "average"), default="first")
    convert_parser.add_argument("--verify", action="store_true")
    verify_parser = subparsers.add_parser("verify", help="Verify persisted weights and full Hugging Face forward")
    verify_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify_parser.add_argument("--source", type=Path, help="Override the source recorded in manifest.json")
    for p in (convert_parser, verify_parser):
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--shapes", default="16x16,12x20,20x12,24x24,1x1",
                       help="Patch grid sizes for verification, including down/up sampling and padding")
        p.add_argument("--atol", type=float, default=5e-4)
        p.add_argument("--rtol", type=float, default=3e-4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    shapes = []
    for token in args.shapes.split(","):
        try:
            height, width = map(int, token.lower().split("x"))
            if min(height, width) <= 0:
                raise ValueError
        except ValueError:
            parser.error(f"Invalid patch grid: {token!r}")
        shapes.append((height, width))
    if args.command == "convert":
        convert(args.source, args.output, args.temporal_patch_size, args.temporal_init)
    if args.command == "verify" or args.verify:
        verify(args.output, source=args.source, device=args.device, shapes=shapes, atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()
