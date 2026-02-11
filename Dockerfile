# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Install curl for downloading models from HuggingFace
RUN apk add --no-cache curl

# Install dependencies
COPY package*.json ./
RUN npm install --no-audit --no-fund

# Copy source files
COPY . .

# Cache-busting arg — change to force model re-download
ARG MODEL_VERSION=v1

# --- Download models from HuggingFace ---

# Anamnesis (pixel-space)
RUN echo "Model version: $MODEL_VERSION" && \
    mkdir -p public/models/anamnesis public/init/anamnesis && \
    curl -fSL -o public/models/anamnesis/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/denoiser.onnx" && \
    curl -fSL -o public/init/anamnesis/init_state.json \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/init_state.json"

# Anamnesis Latent
RUN mkdir -p public/models/anamnesis_latent public/init/anamnesis_latent && \
    curl -fSL -o public/models/anamnesis_latent/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/latent/denoiser.onnx" && \
    curl -fSL -o public/models/anamnesis_latent/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/latent/decoder.onnx" && \
    curl -fSL -o public/init/anamnesis_latent/init_state.json \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/latent/init_state.json"

# Anamnesis Consistency (1-step)
RUN mkdir -p public/models/anamnesis_consistency public/init/anamnesis_consistency && \
    curl -fSL -o public/models/anamnesis_consistency/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/consistency/denoiser.onnx" && \
    curl -fSL -o public/models/anamnesis_consistency/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/consistency/decoder.onnx" && \
    curl -fSL -o public/init/anamnesis_consistency/init_state.json \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/consistency/init_state.json"

# Anamnesis Small (~20M params)
RUN mkdir -p public/models/anamnesis_small public/init/anamnesis_small && \
    curl -fSL -o public/models/anamnesis_small/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/small/denoiser.onnx" && \
    curl -fSL -o public/models/anamnesis_small/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/small/decoder.onnx" && \
    curl -fSL -o public/init/anamnesis_small/init_state.json \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/small/init_state.json"

# Anamnesis Tiny (~10M params)
RUN mkdir -p public/models/anamnesis_tiny public/init/anamnesis_tiny && \
    curl -fSL -o public/models/anamnesis_tiny/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/tiny/denoiser.onnx" && \
    curl -fSL -o public/models/anamnesis_tiny/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/tiny/decoder.onnx" && \
    curl -fSL -o public/init/anamnesis_tiny/init_state.json \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/tiny/init_state.json"

# Anamnesis Small Distill (1-step consistency)
RUN mkdir -p public/models/anamnesis_small_distill public/init/anamnesis_small_distill && \
    curl -fSL -o public/models/anamnesis_small_distill/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/small_distill/denoiser.onnx" && \
    curl -fSL -o public/models/anamnesis_small_distill/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/small_distill/decoder.onnx" && \
    curl -fSL -o public/init/anamnesis_small_distill/init_state.json \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/small_distill/init_state.json"

# Anamnesis Small Distill FP16
RUN mkdir -p public/models/anamnesis_small_distill_fp16 && \
    curl -fSL -o public/models/anamnesis_small_distill_fp16/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/small_distill_fp16/denoiser.onnx" && \
    curl -fSL -o public/models/anamnesis_small_distill_fp16/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/small_distill_fp16/decoder.onnx"

# Mercury Flow Consistency (1-step)
RUN mkdir -p public/models/mercury_flow_consistency public/init/mercury_flow_consistency && \
    curl -fSL -o public/models/mercury_flow_consistency/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/mercury_flow_consistency/denoiser.onnx" && \
    curl -fSL -o public/models/mercury_flow_consistency/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/mercury_flow_consistency/decoder.onnx" && \
    curl -fSL -o public/models/mercury_flow_consistency/init_state.json \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/mercury_flow_consistency/init_state.json" && \
    curl -fSL -o public/init/mercury_flow_consistency/init_state.json \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/mercury_flow_consistency/init_state.json"

# Mercury Flow Consistency FP16
RUN mkdir -p public/models/mercury_flow_consistency_fp16 && \
    curl -fSL -o public/models/mercury_flow_consistency_fp16/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/mercury_flow_consistency_fp16/denoiser.onnx" && \
    curl -fSL -o public/models/mercury_flow_consistency_fp16/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/mercury_flow_consistency_fp16/decoder.onnx"

# Tube Runner (Mercury Flow HD 512)
RUN mkdir -p public/models/tube_runner public/init/tube_runner && \
    curl -fSL -o public/models/tube_runner/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/tube_runner/denoiser.onnx" && \
    curl -fSL -o public/models/tube_runner/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/tube_runner/decoder.onnx" && \
    curl -fSL -o public/models/tube_runner/init_state.json \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/tube_runner/init_state.json" && \
    curl -fSL -o public/init/tube_runner/init_state.json \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/tube_runner/init_state.json"

# Tube Runner FP16
RUN mkdir -p public/models/tube_runner_fp16 && \
    curl -fSL -o public/models/tube_runner_fp16/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/tube_runner_fp16/denoiser.onnx" && \
    curl -fSL -o public/models/tube_runner_fp16/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/anamnesis-demo-011225/resolve/main/tube_runner_fp16/decoder.onnx"

# Flappy Bird (denoiser only — no upsampler in lite)
RUN mkdir -p public/models/flappy_bird public/init/flappy_bird && \
    curl -fSL -o public/models/flappy_bird/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/flappy-bird-demo/resolve/main/denoiser.onnx" && \
    curl -fSL -o public/init/flappy_bird/init_state.json \
      "https://huggingface.co/datasets/Karajan42/flappy-bird-demo/resolve/main/init_state.json"

# Knightfall 002
RUN mkdir -p public/models/knightfall_002 public/init/knightfall_002 && \
    curl -fSL -o public/models/knightfall_002/denoiser.onnx \
      "https://huggingface.co/datasets/Karajan42/knightfall_demo_002/resolve/main/denoiser.onnx" && \
    curl -fSL -o public/models/knightfall_002/decoder.onnx \
      "https://huggingface.co/datasets/Karajan42/knightfall_demo_002/resolve/main/decoder.onnx" && \
    curl -fSL -o public/models/knightfall_002/init_state.json \
      "https://huggingface.co/datasets/Karajan42/knightfall_demo_002/resolve/main/init_state.json" && \
    curl -fSL -o public/init/knightfall_002/init_state.json \
      "https://huggingface.co/datasets/Karajan42/knightfall_demo_002/resolve/main/init_state.json"

# Copy ORT WASM files to public/wasm
RUN mkdir -p public/wasm && \
    cp node_modules/onnxruntime-web/dist/*.wasm public/wasm/ && \
    cp node_modules/onnxruntime-web/dist/ort-wasm*.mjs public/wasm/

# Build
RUN npm run build

# --- Production stage ---
FROM nginx:alpine

RUN apk add --no-cache gettext

COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen \$PORT default_server;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json application/wasm;

    # Required for SharedArrayBuffer + WebAssembly multi-threading
    add_header Cross-Origin-Embedder-Policy "require-corp" always;
    add_header Cross-Origin-Opener-Policy "same-origin" always;

    location / {
        try_files \$uri \$uri/ /index.html;
    }

    location ~* \\.(?:wasm|mjs)$ {
        types {
            application/wasm wasm;
            text/javascript mjs;
        }
        add_header Cross-Origin-Embedder-Policy "require-corp" always;
        add_header Cross-Origin-Opener-Policy "same-origin" always;
        add_header Cache-Control "public, max-age=31536000, immutable";
    }

    location ~* \\.(?:onnx)$ {
        add_header Cross-Origin-Embedder-Policy "require-corp" always;
        add_header Cross-Origin-Opener-Policy "same-origin" always;
        add_header Cache-Control "public, max-age=86400";
    }

    location ~* \\.(?:css|js|jpg|jpeg|gif|png|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Cross-Origin-Embedder-Policy "require-corp" always;
        add_header Cross-Origin-Opener-Policy "same-origin" always;
    }
}
EOF

COPY --from=builder /app/dist /usr/share/nginx/html

RUN echo '#!/bin/sh' > /docker-entrypoint.sh && \
    echo 'export PORT=${PORT:-8080}' >> /docker-entrypoint.sh && \
    echo 'envsubst '\''$PORT'\'' < /etc/nginx/conf.d/default.conf > /etc/nginx/conf.d/default.conf.tmp' >> /docker-entrypoint.sh && \
    echo 'mv /etc/nginx/conf.d/default.conf.tmp /etc/nginx/conf.d/default.conf' >> /docker-entrypoint.sh && \
    echo 'nginx -g "daemon off;"' >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh

EXPOSE 8080

CMD ["/docker-entrypoint.sh"]
