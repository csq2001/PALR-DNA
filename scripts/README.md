# Pipeline entry points

The supported workflow is:

```text
prepare_wikiart_face.py
  -> ../train_AE.py
  -> cache_latents.py
  -> ../train_DLRT.py
  -> encode_image_to_dna.py
  -> inject_dna_errors.py
  -> correct_dna_to_latent.py
  -> reconstruct_image.py
```

Default trained-model paths:

```text
outputs/checkpoints/AE.pth
outputs/checkpoints/DLRT.pth
```

See the project-level `README.md` for complete commands.
