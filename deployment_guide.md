# 🚀 Deploying Aura Content Supply Chain to Render (Unified Single-Service)

This deployment guide details how to configure, deploy, and package the unified **Aura Content Supply Chain Cockpit** as a single service on **Render** (FastAPI backend natively serving the static HTML frontend SPA).

---

## 🐍 Unified Render Deployment Steps

Render is ideal for running persistent Python APIs. Because our FastAPI server is configured to serve the frontend static assets directly on the root `/` path, we do not need separate hosting on Vercel. 

### 1. Repository Configuration
Ensure your GitHub repository has the correct structure (matching our workspace layout):
```
aura-content-supply-chain/
├── backend/
│   ├── main.py
│   └── requirements.txt
└── frontend/
    └── index.html
```

### 2. Set Up Web Service on Render
1. Log in to the [Render Dashboard](https://dashboard.render.com) and click **New > Web Service**.
2. Connect your GitHub repository containing the supply chain code.
3. Configure the following parameters during creation:
   - **Name**: `aura-content-supply-chain`
   - **Runtime**: `Python 3`
   - **Branch**: `main` (or your active working branch)
   - **Root Directory**: `backend` (Crucial! This scopes the build to the backend folder)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 3. Environment Variables
To authenticate with the Gemini API safely:
1. In the Web Service configuration on Render, go to the **Environment** tab.
2. Add a new Environment Variable:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: `YOUR_SECRET_GEMINI_API_KEY_HERE`
3. Save changes. Render will automatically trigger a secure rebuild and deploy!

---

## 🛡️ Production Verification & CORS-Free Routing

Because both the frontend SPA and the backend orchestrator run on the exact same domain (e.g. `https://aura-content-supply-chain.onrender.com`), the application natively bypasses **Cross-Origin Resource Sharing (CORS)** blocks. 

* The HTML frontend is served on the root `/` endpoint.
* The API dispatches run on relative paths `/api/v1/orchestrate`.
* Telemetry indicators check `/health` on startup.

This ensures lightning-fast execution, zero-configuration routing, and maximum security for enterprise demonstrations.
