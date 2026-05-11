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

### L4 + NF4 (default) and Hugging Face access

The GPU image sets **`TALKIE_QUANTIZATION=nf4`**: the loader builds the module graph on the **meta** device and uses **`load_state_dict(..., assign=True)`** so the ~26 GiB checkpoint is **not** duplicated alongside a full random CPU init (which OOM’d Cloud Run’s 32 GiB RAM). Linear layers are then moved to **NF4** on the GPU block-by-block. The **embedding** and **`lm_head`** stay **bf16** (~1.3 GiB combined for a 64k vocab). **Quality** may be slightly lower than full bf16; set **`TALKIE_QUANTIZATION=none`** on a larger GPU if you need the original path.

**Hugging Face Hub:** the first cold start still downloads the checkpoint into **`HF_HOME`** (ephemeral on Cloud Run unless you add a volume). Set a read token so you are not throttled and can access private repos:

- Env **`HF_TOKEN`** or **`HUGGING_FACE_HUB_TOKEN`** (both are honored by `huggingface_hub`), or  
- **`gcloud run deploy ... --set-secrets=HF_TOKEN=hf-token:latest`** after creating a Secret Manager secret.

Add **`--set-secrets`** on the **`gcloud run deploy`** line below when you use a secret (see the full example).

### GPU region (try EU if `us-central1` quota is stuck)

Per [Cloud Run GPU regions](https://cloud.google.com/run/docs/configuring/services/gpu), **NVIDIA L4** is offered in a small set of locations. **`us-central1`**, **`us-east4`**, and **`asia-south1`** are documented as **invitation-only** for some customers; **`europe-west1`** (Belgium) and **`europe-west4`** (Netherlands) are the usual **first alternatives** when Iowa/Virginia won’t grant quota.

You can run **`talkie-web`** in **`us-central1`** (custom domain, low latency for browsers) and **`talkie-gpu`** in **another region**. The web service calls the GPU URL over HTTPS; expect **extra cross-region latency** on each chat request, which is often acceptable for this workload.

- **GitHub:** workflow **Deploy Talkie GPU** → **Run workflow** → set **region** (e.g. `europe-west4`) instead of the default `us-central1`. The workflow still updates **`talkie-web`** in `us-central1` with **`TALKIE_UPSTREAM_URL`** pointing at the GPU service’s regional `*.run.app` URL.
- **CLI / quota:** Each region has its own **L4 quota state**. If you have no preference yet for that region, use **`gcloud beta quotas preferences create`** with `--dimensions=region=europe-west4` (and a **new** `--preference-id`). If deploy still fails, use the [quotas console](https://console.cloud.google.com/iam-admin/quotas) filtered to **Cloud Run** and **NVIDIA L4** for that region.

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
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 32Gi \
  --cpu 8 \
  --timeout 3600 \
  --min-instances 0 \
  --max-instances 1 \
  --port 8080 \
  --set-env-vars "TALKIE_MODEL_NAME=talkie-1930-13b-it,TALKIE_QUANTIZATION=nf4,HF_HOME=/tmp/hf"
```

`--no-gpu-zonal-redundancy` uses a **different Cloud Run GPU quota bucket**. If **`gcloud run deploy`** fails with *no quota for GPUs **without** zonal redundancy*, **drop** that flag (default in the **Deploy Talkie GPU** GitHub Action). If it fails for *with* zonal redundancy instead, request **GPU quota** for your project: [g.co/cloudrun/gpu-quota](https://g.co/cloudrun/gpu-quota). The workflow **Run workflow** form includes a checkbox to pass `--no-gpu-zonal-redundancy` when your project only has that pool.

### Request L4 quota from the CLI (optional)

Use [**`gcloud beta quotas preferences create`**](https://cloud.google.com/docs/quotas/api) so Google can reach you. Castalia’s contact for quota correspondence:

**`custodian@castalia.institute`**

Replace `REGION` if you deploy outside `us-central1`. Use a unique `--preference-id` on **create** only if no preference exists yet for that quota + region.

**If `create` fails with** `Quota Preference with dimension '{}' already exist` — Google allows **one** preference per (`service`, `quota-id`, `dimensions`) on the project. List what you already have, then call **`update`** with that preference **id** (the last path segment of `name`):

```bash
gcloud beta quotas preferences list --project="$PROJECT_ID" \
  --format="table(name.basename(),quotaId,dimensions)"
```

Example **update** (same flags as create, but positional id + **`preferences update`**):

```bash
gcloud beta quotas preferences update talkie-nvidia-l4-us-central1-zonal \
  --project="$PROJECT_ID" \
  --service=run.googleapis.com \
  --quota-id=NvidiaL4GpuAllocPerProjectRegion \
  --dimensions=region="$REGION" \
  --preferred-value=1 \
  --email=custodian@castalia.institute \
  --justification='Cloud Run talkie-gpu: 1x NVIDIA L4 for Talkie 13B inference (Castalia Institute).'
```

Use your real preference id from `list` (the example id matches an existing Castalia project entry; substitute if yours differs). Do the same pattern for **`NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion`** using its preference id.

**Zonal redundancy pool** (default Cloud Run GPU deploy):

```bash
export REGION=us-central1
export PROJECT_ID=$(gcloud config get-value project)

gcloud beta quotas preferences create \
  --project="$PROJECT_ID" \
  --service=run.googleapis.com \
  --quota-id=NvidiaL4GpuAllocPerProjectRegion \
  --dimensions=region="$REGION" \
  --preferred-value=1 \
  --email=custodian@castalia.institute \
  --justification='Cloud Run talkie-gpu: 1x NVIDIA L4 for Talkie 13B inference (Castalia Institute).' \
  --preference-id="castalia-talkie-l4-${REGION}-zonal"
```

**No zonal redundancy pool** (only if you deploy with `--no-gpu-zonal-redundancy`):

```bash
gcloud beta quotas preferences create \
  --project="$PROJECT_ID" \
  --service=run.googleapis.com \
  --quota-id=NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion \
  --dimensions=region="$REGION" \
  --preferred-value=1 \
  --email=custodian@castalia.institute \
  --justification='Cloud Run talkie-gpu: 1x L4, no zonal redundancy pool.' \
  --preference-id="castalia-talkie-l4-${REGION}-nozonal"
```

The address must be a Google account **allowed to approve quota** on this project, or Google may reject follow-ups.

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

The **`web/`** container must include **`requests`** (declared in **`web/requirements.txt`**) so `google.oauth2.id_token.fetch_id_token` can run; without it, the proxy may forward **without** a Bearer token and the GPU returns **403**.

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

(Or paste the URL manually.)

**GitHub Actions:** the **`Deploy Talkie GPU`** workflow sets **`TALKIE_UPSTREAM_URL` on the Cloud Run service** after a successful GPU deploy, so you often **do not** need a repo secret. Repository secret **`TALKIE_UPSTREAM_URL`** is optional: if set, **`Deploy Cloud Run`** passes **`--update-env-vars`** so redeploys from `main` can override the URL. The default **`GITHUB_TOKEN`** cannot create or update other secrets by itself.

**Optional — mirror the URL into a repo Actions secret:** add repository secret **`GH_ACTIONS_SECRETS_TOKEN`**, a [fine-grained personal access token](https://github.com/settings/personal-access-tokens/new) for this repo only, with **Secrets and variables → Actions: Read and write**. After each successful **`Deploy Talkie GPU`** run, the workflow runs `gh secret set TALKIE_UPSTREAM_URL` so **`TALKIE_UPSTREAM_URL`** stays in sync for **`Deploy Cloud Run`**. Omit **`GH_ACTIONS_SECRETS_TOKEN`** if you do not want a PAT in the repository.
