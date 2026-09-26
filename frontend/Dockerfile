FROM node:22-alpine AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1 \
    BACKEND_URL=http://backend:8000
COPY package.json package-lock.json ./
RUN npm ci
COPY . ./
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    BACKEND_URL=http://backend:8000
COPY --from=builder --chown=node:node /app ./
USER node
EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
    CMD node -e "require('http').get('http://127.0.0.1:3000/login', r => process.exit(r.statusCode === 200 ? 0 : 1)).on('error', () => process.exit(1))"
CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0"]
