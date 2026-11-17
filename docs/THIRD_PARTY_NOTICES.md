# Source attribution and licenses

The root LICENSE applies to the project application code. Upstream source and model terms continue to apply to their respective components.

- `integrations/genwarp_local/genwarp/` contains the GenWarp implementation used for generated views, from [Sony GenWarp](https://github.com/sony/genwarp). Its LICENSE and NOTICE are retained alongside the source. The local camera helper is included as `integrations/genwarp_local/extract_latents.py`.
- `integrations/splatting/` contains the CUDA/CPU splatting implementation from [pesser/splatting](https://github.com/pesser/splatting), with its LICENSE. Build products are excluded and must be compiled in the target CUDA environment.
- ZoeDepth and MiDaS are acquired from their upstream repositories. Their source and pretrained weight terms apply separately. GenWarp's pretrained image encoder, VAE and generator weights must also be obtained from their providers.
- CosPlace and EigenPlaces checkpoints are acquired from their respective official releases. ImageNet reference weights are acquired from PyTorch.

`docs/MODULE_PROVENANCE.json` records the input and application source hashes for the functional refactor. The retrieval, camera and statistical calculations are preserved; imports, runtime directories, source hashing and configuration handling are adapted for the application package.

Datasets and pretrained weights are not redistributed in this repository ZIP. File hashes identify supplied source bytes; they do not authenticate separately downloaded model weights on behalf of their publishers.
