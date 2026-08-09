# VRF/VRV AI Technical Assistant — Frontend

React + Vite frontend for the VRF/VRV technical chatbot. See the root
[`CLAUDE.md`](../../CLAUDE.md) for the overall project and the agent
access-boundary rules this directory is developed under.

## Design & API source of truth

- UI/UX spec: `Documentation/ui-ux-design/` (`01-design-system.md`,
  `02-login-page.md`, `03-app-shell-navigation.md`).
- API contract: `Documentation/system-design/08-authentication-rbac.md` §8.5
  (auth), and other `system-design/` docs as features land.

Design tokens are implemented as CSS custom properties in
`src/styles/tokens.css` using the exact token names from
`01-design-system.md` — do not rename without updating that document first.

## Getting started

```bash
npm install
cp .env.example .env   # set VITE_API_BASE_URL if not using the default
npm run dev
```

## Scripts

| Script | Purpose |
|---|---|
| `npm run dev` | Vite dev server |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Preview the production build locally |
| `npm run lint` | oxlint |
| `npm run test` | Vitest (unit/component tests) |

## Environment variables

See `.env.example`. All variables consumed by the app must be prefixed
`VITE_` (Vite requirement) and are documented in `src/lib/env.js`.
`VITE_API_BASE_URL` is inlined into the JS bundle **at build time** — see
the note in `Dockerfile` if building a container image.

## Architecture notes

- `src/styles/` — design tokens, custom scrollbar (§6), global resets.
- `src/components/` — reusable design-system components (`Button`, `Input`,
  `AIPulseLoader`).
- `src/hooks/` — `usePrefersReducedMotion` (global motion-preference flag,
  §5.4) and `useScrollbarIdle` (custom scrollbar idle-fade behavior, §6).
- `src/lib/httpClient.js` — fetch wrapper with the refresh-on-401
  interceptor pattern (`POST /auth/refresh`, `credentials: "include"`,
  deduped across concurrent 401s).
- `src/auth/` — `AuthContext` (in-memory access token via `tokenStore.js`,
  never localStorage), `filterNavByScopes.js` (single source of truth for
  menu visibility, used by both `Sidebar` and `RouteGuard`), `RouteGuard`
  (defense-in-depth against direct URL access).
- `src/shell/` — `AppShell`, `Sidebar`, `Topbar`, `UserMenu`.
- `src/pages/LoginPage/` — full login flow (default/loading/401/429 with
  countdown/network error states).

## Docker

Multi-stage `Dockerfile` (Node build → nginx static serve). Build arg
`VITE_API_BASE_URL` must be set at **build time** (see comments in
`Dockerfile`) since Vite inlines env vars into the bundle — it is not a
runtime container env var. Must be built/run via WSL, not Windows-native
Docker (`CLAUDE.md` §5):

```bash
# from WSL
docker build -t vrf-frontend --build-arg VITE_API_BASE_URL=http://localhost:8000/api/v1 .
docker run -p 8080:80 vrf-frontend
```
