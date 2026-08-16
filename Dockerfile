FROM node:20-slim

# Python for the ingestion/detector/scoring/briefs/export pipeline, which
# runs in this same container (shelled out to from the refresh endpoint).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Python dependencies
COPY requirements.txt ./
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# Node dependencies for the Next.js app (lives in web/, not repo root)
COPY web/package.json web/package-lock.json ./web/
RUN npm ci --prefix web

# Application code
COPY . .

# Build Next.js in standalone mode (see web/next.config.ts: output: "standalone")
RUN npm run build --prefix web

# Standalone output omits static assets by design — copy them in manually.
# Next's own build tracer may have already created .next/standalone/public
# (populated only with the specific files server code reads via fs, e.g.
# leaderboard.json) — copying with a trailing `/.` merges into it instead
# of nesting a `public/public/...` subdirectory when the destination
# already exists.
RUN mkdir -p web/.next/standalone/public web/.next/standalone/.next/static \
    && cp -r web/public/. web/.next/standalone/public/ \
    && cp -r web/.next/static/. web/.next/standalone/.next/static/

ENV DB_PATH=/data/anomaly_radar.duckdb
ENV NODE_ENV=production
ENV PORT=3000
# Next's standalone server binds to `localhost` unless told otherwise, which
# is unreachable from outside the container (Railway/any reverse proxy) —
# must bind all interfaces.
ENV HOSTNAME=0.0.0.0
EXPOSE 3000

CMD ["node", "web/.next/standalone/server.js"]
