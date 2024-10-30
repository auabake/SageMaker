FROM public.ecr.aws/sagemaker/sagemaker-distribution:latest-cpu
ARG NB_USER="sagemaker-user"
ARG NB_UID=1000
ARG NB_GID=100
ENV MAMBA_USER=$NB_USER
ARG R_VERSION=4.4.0

USER root

 # install scrapy in the base environment
RUN micromamba install -y --name base -c conda-forge scrapy

 # download VSCodeVim
RUN \
  wget https://github.com/VSCodeVim/Vim/releases/download/v1.27.2/vim-1.27.2.vsix \
  -P /tmp/exts/ --no-check-certificate

 # Install the extension
RUN \
  extensionloc=/opt/amazon/sagemaker/sagemaker-code-editor-server-data/extensions \
  && sagemaker-code-editor \
    --install-extension "/tmp/exts/vim-1.27.2.vsix" \
    --extensions-dir "${extensionloc}"

RUN apt-get --allow-releaseinfo-change update && \
    mkdir -p /usr/share/man/man1 && \
    apt-get -y install openjdk-11-jre-headless default-jre openjdk-11-jdk curl less unzip jq samba-client

RUN apt-get install -y awscli

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    SHELL=/bin/bash \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN python3 -m ipykernel install --prefix=/opt/conda --name bhp-python  --display-name "bhp-python"

RUN conda config --set ssl_verify false

RUN pip install --upgrade pip

RUN pip install --upgrade sagemaker

RUN curl -O https://cdn.rstudio.com/r/ubuntu-2204/pkgs/r-${R_VERSION}_1_amd64.deb \
    && apt-get install -yq --no-install-recommends ./r-${R_VERSION}_1_amd64.deb \
    && rm -f ./r-${R_VERSION}_1_amd64.deb \
    && ln -s /opt/R/${R_VERSION}/bin/R /usr/local/bin/R \
    && ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/local/bin/Rscript

RUN chmod a+rwx -R /opt/R \
    && chmod a+rwx -R /usr/local/bin/R \
    && chmod a+rwx -R /usr/local/bin/Rscript

RUN /opt/R/${R_VERSION}/bin/R -e 'install.packages(c( \
        "IRkernel" \
    ), \
        repos="https://cloud.r-project.org/", quiet=TRUE)'

USER $MAMBA_USER

RUN /opt/R/${R_VERSION}/bin/R -e 'IRkernel::installspec()'

RUN mv /home/sagemaker-user/.local/share/jupyter/kernels/ir /opt/conda/share/jupyter/kernels

RUN sed -i 's/"display_name": "R"/"display_name": "bhp-r"/' /opt/conda/share/jupyter/kernels/ir/kernel.json

USER root

RUN /opt/R/${R_VERSION}/bin/R -e 'install.packages(c( \
        "https://cran.r-project.org/src/contrib/Archive/corpus/corpus_0.10.2.tar.gz" \
    ), \
    repos = NULL, type = "source", dependencies = TRUE, quiet=TRUE)'

RUN apt-get update -y && \
    apt-get upgrade -y

RUN chmod -R 777 /opt/R && \
    chmod -R 777 /tmp

USER $MAMBA_USER
ENTRYPOINT ["entrypoint-code-editor"]