# Deploy Talkie web (Cloud Run)

This repository is primarily a **local inference library** for 13B models. The `web/` app is a tiny FastAPI landing page suitable for **Google Cloud Run** (bill when handling requests; scale to zero).

## One-time

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

## Build and deploy

From the repository root:

```bash
export REGION=us-central1
export SERVICE=talkie-web

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --concurrency 80
```

## Custom domain (`talkie.castalia.institute`)

1. Create a domain mapping (replace project/region if needed):

```bash
gcloud beta run domain-mappings create \
  --service "$SERVICE" \
  --domain talkie.castalia.institute \
  --region "$REGION" \
  --project YOUR_PROJECT_ID
```

2. Apply the **DNS records** the command prints (often a `TXT` for verification and a `CNAME` to `ghs.googlehosted.com` or the hostname Google shows).

3. Wait for TLS provisioning (can take tens of minutes).

If DNS is on **Cloudflare**, use **DNS only** (grey cloud) for the record Google expects for domain mapping, or follow [Google’s Cloud Run + Cloudflare](https://cloud.google.com/run/docs/mapping-custom-domains) guidance so certificate issuance succeeds.

## DNS for `talkie.castalia.institute`

In the `castalia.institute` zone, add:

| Name   | Type  | Target                 |
|--------|-------|------------------------|
| talkie | CNAME | ghs.googlehosted.com.  |

After DNS propagates, Google will finish TLS certificate provisioning (often 15–60 minutes).

## Public access (`403` from `*.run.app`)

If `gcloud run deploy --allow-unauthenticated` completes but the service URL returns **403**, an **organization policy** may block `allUsers` as Cloud Run invoker. An org admin must allow public invokers for this project (or grant `roles/run.invoker` to specific principals). Until then, use an identity that has invoke permission.

## Deploy Talkie GPU API (real weights)

This fork includes **`gpu_server/`**: a FastAPI service that loads **`talkie.generate.Talkie`** and implements **`POST /v1/chat/completions`** in OpenAI shape so the CPU site can proxy to **actual Talkie inference** (no substitute LLM).

Build the image from the **repo root** (not `web/`):

```bash
docker build -f Dockerfile.gpu -t talkie-gpu .
```

Run locally (CUDA machine):

```bash
docker run --gpus all -e PORT=8080 -p 8080:8080 talkie-gpu
```

The first start downloads weights from Hugging Face into `HF_HOME` (default in container: use `-e HF_HOME=/path/with/space` if you mount a volume).

### VRAM and Cloud Run

The upstream README targets **~28 GiB VRAM** for bfloat16. Cloud Run’s **NVIDIA L4 is 24 GiB** — it may OOM for the full 13B checkpoint; validate on your account or run this API on a larger GPU (GKE, Compute Engine, etc.) and still point `TALKIE_UPSTREAM_URL` at that URL.

### Deploy `talkie-gpu` on Cloud Run (example)

Enable APIs and pick a [GPU region](https://cloud.google.com/run/docs/configuring/services/gpu). Your project needs [**Cloud Run GPU quota**](https://cloud.google.com/run/docs/configuring/services/gpu); if deploy fails with a quota error, request allocation (see [GPU quota](https://g.co/cloudrun/gpu-quota)).

The root **`Dockerfile`** is the CPU web app. Build the GPU image with **`cloudbuild.gpu.yaml`** (uses **`Dockerfile.gpu`**), then deploy that image:

```bash
export REGION=us-central1
export PROJECT_ID=$(gcloud config get-value project)
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/talkie-gpu:latest"

gcloud builds submit --config cloudbuild.gpu.yaml --project "$PROJECT_ID" .

gcloud run deploy talkie-gpu \
  --image "$IMAGE" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --no-allow-unauthenticated \
  --no-gpu-zonal-redundancy \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 32Gi \
  --cpu 8 \
  --timeout 3600 \
  --min-instances 0 \
  --max-instances 1 \
  --port 8080 \
  --set-env-vars "TALKIE_MODEL_NAME=talkie-1930-13b-it,HF_HOME=/tmp/hf"
```

`--no-gpu-zonal-redundancy` avoids an extra quota dimension on some accounts; if Google prompts during deploy, choose the option your project’s quota allows.

**`.dockerignore`:** must **not** exclude `src/`, `pyproject.toml`, or `README.md`, or the GPU image build will fail (`Dockerfile.gpu` copies them).

If Hugging Face needs a token, create a Secret Manager secret and add e.g. `--set-secrets HF_TOKEN=hf-token:latest` (and set `HUGGING_FACE_HUB_TOKEN` or download auth per HF docs). Increase **`--timeout`** for first-boot model download; consider **`--min-instances 1`** after debugging cold starts.

After deploy, note the service **URL** and grant the **CPU service’s** runtime identity **`roles/run.invoker`** on **`talkie-gpu`**, as below.

## Chat UI → GPU inference

The `web/` service serves a **circa-1931** chat page at `/` and proxies `POST /v1/chat/completions` to **`gpu_server`** (or any compatible OpenAI-shaped GPU backend).

Set on **`talkie-web`** (CPU):

| Env | Meaning |
|-----|---------|
| `TALKIE_UPSTREAM_URL` | Base URL of the Talkie GPU service (e.g. `https://talkie-gpu-xxxxx.us-central1.run.app`) — **no** trailing slash |
| `TALKIE_UPSTREAM_BEARER` | Optional static bearer if identity tokens are not used |
| `TALKIE_INPUT_USD_PER_1K` | Optional. USD per **1,000 prompt tokens** (injected into the page for footer **est.** cost) |
| `TALKIE_OUTPUT_USD_PER_1K` | Optional. USD per **1,000 completion tokens** for the same footer estimate |

If the rates are unset, the footer still **tallies tokens** from the GPU `usage` object when present; USD appears after you set the rates and redeploy **`talkie-web`**.

If `TALKIE_UPSTREAM_URL` is unset, the UI loads but chat returns **503** (“apparatus is not yet connected”).

On GCP without `TALKIE_UPSTREAM_BEARER`, the CPU service uses **`fetch_id_token`**; grant **talkie-web’s** runtime service account **`roles/run.invoker`** on **`talkie-gpu`**.

```bash
# Replace with your talkie-web service account if customized.
WEB_SA="584409871588-compute@developer.gserviceaccount.com"

gcloud run services add-iam-policy-binding talkie-gpu \
  --region "$REGION" \
  --member "serviceAccount:${WEB_SA}" \
  --role roles/run.invoker \
  --project "$PROJECT_ID"
```

Then point the web service at the GPU URL (and redeploy the web app from the default **`Dockerfile`**, not `Dockerfile.gpu`):

```bash
GPU_URL=$(gcloud run services describe talkie-gpu --region "$REGION" --format='value(status.url)')

gcloud run deploy talkie-web --source . --region "$REGION" \
  --set-env-vars "TALKIE_UPSTREAM_URL=${GPU_URL}"
```

(Or paste the URL manually.) **GitHub Actions:** set repository secret `TALKIE_UPSTREAM_URL` to the GPU base URL so the deploy workflow can pass `--set-env-vars` on each **`talkie-web`** rollout.
