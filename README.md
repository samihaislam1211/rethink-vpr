# Generated-view augmentation for visual place recognition

A complete implementation of database-side and query-side GenWarp augmentation for visual place recognition. The application includes real-image baselines, camera-controlled synthesis, conventional augmentation controls, global retrieval, descriptor/fusion ablations, confidence gating, paired statistical tests and result plots.

## Repository layout

```text
nvs-vpr-eval/
├── README.md
├── run.py                    # configure, check and submit workloads
├── pyproject.toml
├── configs/                  # paths, model/camera settings and source selections
├── requirements/             # separate VPR and GenWarp dependency recipes
├── src/nvs_vpr/
│   ├── baselines/            # descriptors, ground truth and real-image retrieval
│   ├── generation/           # GenWarp, camera protocol and strict cache loading
│   ├── database/             # reference-view synthesis, indexing and evaluation
│   ├── query/                # query synthesis, controls and practical fusion
│   ├── analysis/             # robustness, gates, statistics and plotting
│   ├── cli.py
│   └── pipeline.py
├── scripts/                  # Slurm workflow driver and repository packaging
├── tests/
├── integrations/             # GenWarp/camera helpers and CUDA splatting source
└── docs/                     # source provenance and third-party notices
```

Download the public datasets, pretrained weights, ZoeDepth and MiDaS separately using the instructions below. They are not included in the repository ZIP.

## 1. Requirements and execution model

Use Linux, Python 3.10, an NVIDIA GPU and a compatible driver. The recorded GPU was a V100S with 32 GB VRAM. This implementation submits computation through Slurm; it does not bypass allocation checks. Configure your own partition and GPU resource name. Up to six GPU tasks may run concurrently; fewer are supported. CPU jobs request no GPUs. Allow substantial scratch storage for datasets, generated images and full score tensors; these are created during execution, not supplied in the ZIP.

Two environments keep the verified VPR and older GenWarp dependencies separate:

| Environment | Recorded core versions | Used for |
|---|---|---|
| `vpr` | Python3.10; torch2.7.1+cu126; torchvision0.22.1; NumPy2.2.6 | Descriptors, retrieval, statistics, plots |
| `genwarp` | Python3.10; torch2.0.1+cu118; torchvision0.15.2; NumPy1.26.4; diffusers0.24.0; transformers4.36.2 | GenWarp/ZoeDepth synthesis |

Complete observed package versions are in `configs/{vpr,genwarp}.observed.json`. The bootstrap requirements deliberately use OpenCV4.10 and OmegaConf2.3.0 for GenWarp compatibility; they are installation recipes, not claims that a fresh environment has been tested on every platform. Keep datasets and weights read-only, and use a **new** work directory for each installation/configuration. Identical diffusion pixels across machines are not guaranteed.

## 2. Install the environments

Run from the extracted repository root. Conda is an example; equivalent Python3.10 environments with the same dependencies are acceptable.

```bash
conda create -n vpr python=3.10 -y
conda activate vpr
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/vpr.txt
conda deactivate

conda create -n genwarp python=3.10 -y
conda activate genwarp
python -m pip install 'setuptools<70' wheel
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/genwarp.txt
```

Build the bundled splatting package **inside a GPU allocation**, with a CUDA11.8 toolkit/compiler compatible with torch2.0.1. Its setup script only builds its CUDA extension when a GPU is visible. Set `CUDA_HOME` to your actual toolkit and adjust the Slurm options:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export MAX_JOBS=4
srun --partition=YOUR_PARTITION --gres=gpu:1 --cpus-per-task=4 --mem=8G \
  python -m pip install --no-build-isolation ./integrations/splatting
srun --partition=YOUR_PARTITION --gres=gpu:1 \
  python -c "import torch, splatting.cuda; print(torch.cuda.is_available())"
conda deactivate
```

The root `requirements.txt` includes the VPR requirements. Use the separate GenWarp environment for synthesis. You can run directly from the checkout using `python run.py`; an optional editable installation is `python -m pip install -e . --no-deps`.

## 3. Set up GenWarp, depth estimation and weights

Follow [Sony GenWarp setup](https://github.com/sony/genwarp#how-to-use) for obtaining its models. This repository already includes the local GenWarp source and camera helper used by the experiments under `integrations/genwarp_local/`; the portable runner uses those bytes. An upstream checkout is convenient for its model download script:

```bash
git clone https://github.com/sony/genwarp.git /path/to/genwarp
cd /path/to/genwarp
bash scripts/download_models.sh ./checkpoints
git clone https://github.com/isl-org/ZoeDepth.git extern/ZoeDepth
git clone https://github.com/isl-org/MiDaS.git /path/to/MiDaS
```

Obtain **ZoeD_N**, checkpoint **ZoeD_M12_N.pt**, from the [ZoeDepth model instructions](https://github.com/isl-org/ZoeDepth), and put its absolute path in your configuration. This is the NYU metric-depth variant used here. Do not substitute ZoeD_NK or another depth model for a matched run. MiDaS is loaded locally; inference disables automatic downloads.

The required GenWarp model directory is:

```text
checkpoints/
  multi1/
    config.json
    denoising_unet.pth
    pose_guider.pth
    reference_unet.pth
  image_encoder/
    config.json
    pytorch_model.bin
  sd-vae-ft-mse/
    config.json
    diffusion_pytorch_model.safetensors
```

Use `multi1`, DDIM20, guidance3.5, output512×512. Camera schedules are in `configs/cameras.json`; per-view seeds are `42000000 + 100 × original_source_index + view_index`. The generation adapter records full local hashes before executing. Upstream dependencies may change; run the bounded smoke workflow before large generation. The local GenWarp and splatting snapshot manifests record their source provenance.

## 4. Obtain the four datasets

Acquire data from the linked providers and preserve their access conditions. No dataset is redistributed here.

| Dataset | Access | Exact subset for the configured benchmarks |
|---|---|---|
| GardensPoint | [VPR Tutorial dataset loader](https://github.com/stschubert/VPR_Tutorial/blob/main/datasets/load_dataset.py) links to `https://www.tu-chemnitz.de/etit/proaut/datasets/GardensPoint_Walking.zip` | `day_right` reference200; `night_right` query200 |
| StLucia | Same [upstream loader](https://github.com/stschubert/VPR_Tutorial/blob/main/datasets/load_dataset.py), `https://www.tu-chemnitz.de/etit/proaut/datasets/StLucia_small.zip` | `100909_0845` reference200; `180809_1545` query200; original `GT.npz` |
| Pittsburgh30k | Request dataset access through the [NetVLAD dataset page](https://www.di.ens.fr/willow/research/netvlad/#datasets); then follow [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) and `format_pitts30k.py` | Test database10000; test queries6816 |
| Tokyo24/7 | Request access through the same [dataset provider](https://www.di.ens.fr/willow/research/netvlad/#datasets); use the [formatter instructions](https://github.com/gmberton/VPR-datasets-downloader) and `format_tokyo247.py` | Test database75984; test queries315 |

The full StLucia benchmark is not interchangeable with `StLucia_small`. Preserve the small subset's **original hard and soft labels**:165 hard-eligible queries,200 soft-eligible queries. Do not regenerate soft labels from hard labels. Pittsburgh/Tokyo filenames must retain the standardized `@UTM_east@UTM_north@...` fields, and `.jpg` filenames are sorted case-sensitively. The runner checks counts and database-source identities rather than silently selecting a new subset.

Pittsburgh/Tokyo use25m primary and50m sensitivity GT, global image-level retrieval, and no spatial NMS. Query failure selection uses50m; any-view recovery is separately evaluated at25m/50m. Radius offsets in the camera schedules are not independently calibrated physical translations.

## 5. Obtain descriptor checkpoints

Place these four files in a directory of your choice and configure `descriptor_checkpoints`:

| File | Source |
|---|---|
| `ResNet50_2048_cosplace.pth` | [CosPlace official releases](https://github.com/gmberton/CosPlace/releases), ResNet50/2048D |
| `ResNet50_2048_eigenplaces.pth` | [EigenPlaces official releases](https://github.com/gmberton/EigenPlaces/releases), ResNet50/2048D |
| `alexnet-owt-7be5be79.pth` | [PyTorch ImageNet AlexNet weights](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) |
| `resnet50-11ad3fa6.pth` | [PyTorch ImageNet ResNet50 V2 weights](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) |

`configs/checkpoint_sources.json` retains the nominal original URLs. Local full SHA256 and strict state-dictionary loading establish the weights actually used in each new run. AlexNet-GeM is256D; the other three are2048D. RGB shorter-side480, ImageNet normalization, float32 L2 descriptors and the verified pooling definitions are preserved.

## 6. Configure a new run

```bash
cp configs/project.example.json project.local.json
# Edit every /absolute/path/... value, CUDA_HOME and Slurm partition/GRES.
python3 run.py --config project.local.json check
python3 run.py --config project.local.json setup
```

Set `vpr_python` and `genwarp_python` to the absolute Python executables for the two environments. Set `cuda_home` to the GenWarp CUDA toolkit. The dataset and checkpoint paths can point anywhere accessible to your cluster jobs.

`work_dir` must be a **new directory**. Setup creates the run folders and resolved configuration, records application/configuration hashes, and links the GenWarp source and external weights. It never copies datasets or modifies their contents. Source code stays in `src/nvs_vpr/`; outputs go into the configured work directory. Setup submits no jobs. After setup, changing source/configuration requires a new run directory, so cached outputs remain tied to the code that produced them.

## 7. Run a bounded smoke check first

From the repository root, use a persistent terminal or detach this lightweight scheduler driver:

```bash
nohup bash scripts/run_pipeline.sh project.local.json smoke > smoke-driver.log 2>&1 &
```

It submits metadata preparation, local model hashing, two-image baseline smoke checks and four GenWarp smoke images. Every computational stage runs in Slurm. It records actual job IDs, logs and failures; it stops on a failed stage and does not automatically resubmit. No full augmentation campaign is submitted in `smoke` mode.

Monitor using the job IDs in `work_dir/SUBMISSIONS.json`:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
cat /path/to/new-vpr-run/GENWARP_SMOKE.json
ls /path/to/new-vpr-run/stage_status/
```

Native CUDA build/import failures belong to the GenWarp environment; missing-assets reports identify exact expected paths. Dataset count/GT/source-ID or cache mismatch failures should be investigated before proceeding. Do not retune settings to make old paper numbers match.

## 8. Run the complete experiment workflow

After inspecting the smoke outputs, launch the explicit full mode using the same prepared workspace:

```bash
nohup bash scripts/run_pipeline.sh project.local.json full > full-driver.log 2>&1 &
```

The driver rechecks the bounded smoke gates, then executes these stages in order. All computation is scheduler-allocated, arrays have bounded concurrency, and each completed task's accounting is checked before proceeding:

| Stage | What runs |
|---|---|
| Baselines | Four descriptors × four datasets:16 jobs |
| Database preparation | Exact selected source IDs:GP200,St200,Pitts320 per schedule,Tokyo2392; full B0 reranking check |
| Database pilot | Bounded generation, extraction and end-to-end validation |
| Database augmentation | Newly generated views, E8/Q8 schedules; B=0,1,2,4,8; global original+synthetic index; full and separate Pittsburgh60 cohort |
| Query preparation/pilot | Original IDs/GT;50m baseline failures; bounded camera/depth/feature checks |
| Query augmentation | Fresh GenWarp, eight conventional null views, and eight near-identity views on the selected Pittsburgh failures; CosPlace/max evaluation |
| Robustness | Reuse generated images; add EigenPlaces/AlexNet-GeM/ResNet50-GeM; full max/mean/conf retrieval over original real references |
| Gates/statistics | Original-only signals, frozen200-quantile thresholds, split-half/train-only/twofold decisions, exact McNemar and per-family Holm at25m/50m |

Budgets mean views **per selected original reference image**, not per location or all references. All original queries/references remain; sources may be selected geographically but search is global. Generated images/caches are retained as research artifacts. Full similarity caches can be large; reduce job concurrency if memory/storage/I/O is constrained.

The full driver stops on failures and does not automatically resume already completed workflow steps. After diagnosing a failure, submit only the remaining stages, or prepare a new work directory. Generation and descriptor stages retain their own provenance-aware resume checks.

Individual stages can also be submitted explicitly:

```bash
python3 run.py --config project.local.json submit baseline --array 0-15
python3 run.py --config project.local.json submit database-generate --array 0-5 --shards 6
python3 run.py --config project.local.json submit robust-evaluate --array 0-7
```

These examples are **not independent entry points**: run their prerequisites first. The driver contains the complete sequence. Each submission requests one GPU per GPU task, with concurrency capped by configuration and currently queued allocations; CPU-only arrays request no GPUs. No job is submitted merely by extracting the archive or running `prepare`.

## 9. Outputs

All output is under `work_dir`, separated by function:

```text
work_dir/
├── config/project.json       # resolved run configuration
├── manifests/                # original identities and ordered image paths
├── baselines/results/        # real-image descriptors and Recall@K results
├── generation/runs/          # bounded GenWarp checks and camera records
├── database/results/         # augmentation budgets, pollution, complete rankings
├── query/results/            # max fusion, controls, any-view recovery
├── analysis/
│   ├── results/              # descriptor/fusion/GT grid and Holm corrections
│   ├── gates/                # signals, thresholds, curves and held-out decisions
│   └── figures/              # generated result plots plus source-file mapping
├── logs/                     # persistent Slurm logs
├── stage_status/             # actual task completion/failure records
├── SUBMISSIONS.json          # submitted commands, job IDs and exit information
├── SOURCE_SHA256.json        # application/configuration identity
├── QUERY_RESULTS.csv         # full query-side numerical grid
└── SUMMARY.md
```

The full workflow generates `query_fusion.png`, `confidence_gates.png` and `database_budgets.png` from the current run's saved results. Their source files are recorded in `analysis/figures/SOURCES.json`. Plotting can also be submitted explicitly after its inputs are complete:

```bash
python3 run.py --config project.local.json submit plots
```

The application generates new images and computes every reported metric from that run. It does not require internal audit folders, an author manuscript, or private historical generated images. Historical paper values are not substituted into new results. Any-view recovery, practical fused retrieval and the fixed-prediction oracle are reported separately.

## 10. Test and package the repository

```bash
# In the VPR environment, from the repository root:
python -B -m unittest discover -s tests -v
bash -n scripts/run_pipeline.sh
python scripts/package_repo.py
```

The export is `distributions/NVS_VPR_REPOSITORY.zip`, containing the application source, setup files, configurations, scripts, tests, GenWarp/camera helpers and splatting source. Internal experiment records, old development workflows, manuscripts, datasets, weights, generated images, caches, personal settings and Git metadata are excluded.

After extracting the ZIP, check its integrity with:

```bash
python scripts/verify_repo.py .
```

See [third-party notices](docs/THIRD_PARTY_NOTICES.md) for attribution and licenses. The distribution validation report records the tests actually run. Creating the ZIP does not reinstall dependencies or rerun the GPU experiments.
