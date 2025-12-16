FROM mcr.microsoft.com/devcontainers/base:noble

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Essential packages
    curl tzdata

# Set Timezone
RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Resolve Latest Vespa CLI Version & Install
RUN echo "Resolving latest Vespa CLI version..." &&\

    # 1. get latest Vespa CLI version from GitHub releases/latest page (e.g. .../tag/v8.618.24)
    export LATEST_URL=$(curl -Ls -o /dev/null -w %{url_effective} https://github.com/vespa-engine/vespa/releases/latest) && \

    # 2. extract tag (v8.x.x) from the redirect URL
    export TAG=$(echo $LATEST_URL | awk -F'/' '{print $NF}') && \

    # 3. extract version number from the tag (e.g. 8.618.24) -> for file name combination
    export VERSION=${TAG#v} && \

    # 4. check current architecture (amd64 or arm64)
    export ARCH=$(dpkg --print-architecture) && \

    # 5. download and install
    echo "Downloading Vespa CLI ${VERSION} for ${ARCH}..." && \

    curl -L -o vespa-cli.tar.gz "https://github.com/vespa-engine/vespa/releases/download/${TAG}/vespa-cli_${VERSION}_linux_${ARCH}.tar.gz" && \
    tar -xzf vespa-cli.tar.gz && \

    mv vespa-cli_${VERSION}_linux_${ARCH}/bin/vespa /usr/local/bin/vespa && \
    rm -rf vespa-cli.tar.gz vespa-cli_*

RUN vespa version

USER vscode

RUN vespa config set target http://vespa:8080