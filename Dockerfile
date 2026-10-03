# DynaLM server image (Linux, amd64 or arm64; SIMD tier picked at runtime).
#
#   docker build -t dynalm .
#   docker run --rm -p 8000:8000 -v "$PWD/models:/models" dynalm serve /models/model.gguf
#   docker buildx build --platform linux/amd64,linux/arm64 -t dynalm .   # multi-arch
#
# The server listens on 0.0.0.0 inside the container (DYNALM_HOST); publish
# the port with -p. Models are mounted, never baked into the image.

FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt CMakePresets.json README.md CHANGELOG.md ./
COPY cmake cmake
COPY src src
COPY docs docs
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_TESTS=OFF -DENABLE_BENCHMARKS=OFF -DDYNALM_STATIC_RUNTIME=ON \
    && cmake --build build \
    && cmake --install build --prefix /opt/dynalm

FROM ubuntu:24.04
COPY --from=build /opt/dynalm /opt/dynalm
ENV PATH=/opt/dynalm/bin:$PATH \
    DYNALM_HOST=0.0.0.0 \
    DYNALM_MODELS_DIR=/models
RUN useradd --system --uid 10001 dynalm
USER dynalm
VOLUME /models
EXPOSE 8000
ENTRYPOINT ["dynalm"]
CMD ["--help"]
