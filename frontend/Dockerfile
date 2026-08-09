# syntax=docker/dockerfile:1

# ---- Build stage -----------------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# Vite inlines VITE_-prefixed env vars into the JS bundle AT BUILD TIME (they
# are NOT read at container runtime) — so the backend base URL must be
# supplied as a build ARG, not a `docker run -e` / compose `environment:`
# entry. It must be a URL reachable from the END USER'S BROWSER (fetches run
# client-side, not container-to-container) — see the DIRECT MESSAGE to
# Backend Engineer in the PR/status report for the value to use per
# environment.
ARG VITE_API_BASE_URL=http://localhost:8000/api/v1
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# Install deps first (better layer caching — only re-runs when
# package*.json change).
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---- Runtime stage ----------------------------------------------------------
# Static build output served by nginx — standard, minimal-footprint choice
# for a production SPA bundle (no Node runtime needed at runtime).
FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O- http://127.0.0.1:80/ >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
