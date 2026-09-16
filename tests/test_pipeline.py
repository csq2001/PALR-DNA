from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from core.ae_codec import (
    decode_latent_to_image,
    encode_image_to_latent,
    save_tensor_png,
)
from core.dlrt_decoder import correct_corrupted_records_to_latents
from core.dlrt_model import DNAByteNet
from core.dna_codec import encode_latent_to_dna_records, mutate_dna
from models import AutoEncoder


def test_small_ae_dna_dlrt_pipeline(tmp_path: Path) -> None:
    """Smoke-test every in-memory stage without asserting learned accuracy."""
    device = torch.device("cpu")
    image_path = tmp_path / "input.png"
    Image.new("RGB", (16, 16), color=(48, 128, 208)).save(image_path)

    torch.manual_seed(11)
    ae = AutoEncoder(
        in_channels=3,
        latent_channels=16,
        base_channels=8,
        latent_quant_step=1.0,
    ).eval()
    latent = encode_image_to_latent(ae, image_path, device)
    assert latent.dtype == torch.int8
    assert latent.shape == (16, 2, 2)

    clean_records = list(
        encode_latent_to_dna_records(latent, image_name=image_path.name)
    )
    assert len(clean_records) == 4
    assert all(set(record["dna"]) <= set("ACGT") for record in clean_records)

    corrupted_records = []
    for record in clean_records:
        noisy_dna, errors = mutate_dna(
            record["dna"],
            sub_rate=0.03,
            ins_rate=0.015,
            del_rate=0.015,
            seed=1000 + int(record["position_id"]),
        )
        corrupted_records.append(
            {**record, "noisy_dna": noisy_dna, "errors": errors}
        )
    assert sum(record["errors"]["total"] for record in corrupted_records) > 0
    assert any(
        record["noisy_dna"] != record["dna"] for record in corrupted_records
    )

    torch.manual_seed(17)
    dlrt = DNAByteNet(
        latent_channels=16,
        target_bytes=22,
        max_len=192,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.0,
    ).eval()
    corrected = correct_corrupted_records_to_latents(
        records=corrupted_records,
        model=dlrt,
        model_args={
            "max_len": 192,
            "position_nt": 14,
            "byte_code_nt": 7,
            "byte_codebook_mode": "designed",
            "byte_codebook_seed": 42,
        },
        device=device,
        latent_shape=tuple(latent.shape),
        top_k=2,
        index_top_k=4,
        beam_size=4,
        infer_batch_size=4,
    )

    assert corrected["metrics"]["decoded_chains"] == len(clean_records)
    assert torch.equal(corrected["truth_latent"], latent)
    for key in ("recovered_latent", "top1_completion_latent"):
        assert corrected[key].dtype == torch.int8
        assert corrected[key].shape == latent.shape

    reconstruction = decode_latent_to_image(
        ae,
        corrected["top1_completion_latent"],
        device,
    )
    assert reconstruction.shape == (1, 3, 16, 16)
    assert torch.isfinite(reconstruction).all()
    assert 0.0 <= float(reconstruction.min()) <= float(reconstruction.max()) <= 1.0

    output_path = tmp_path / "reconstruction.png"
    save_tensor_png(reconstruction, output_path)
    with Image.open(output_path) as saved:
        assert saved.size == (16, 16)
        assert saved.mode == "RGB"
