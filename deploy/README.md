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
