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

# Standalone output omits static assets by design. .next/static is
# immutable build output, fine as a one-time copy. public/ is NOT a
# one-time copy: it's where export.publish (run post-deploy, by the
# refresh endpoint) writes fresh leaderboard/entity JSON, and it must
# stay writable-and-visible for the life of the container — so it's a
# symlink back to the source tree, not a copy. (Next's own build tracer
# may have already created a `public/` here containing just the specific
# files server code reads via fs at build time — rm it first so the
# symlink can take its place.)
RUN mkdir -p web/.next/standalone/.next/static \
    && rm -rf web/.next/standalone/public \
    && ln -s /app/web/public web/.next/standalone/public \
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
