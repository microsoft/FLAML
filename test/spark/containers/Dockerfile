ARG PYTHON_VERSION=3.11
FROM eclipse-temurin:17-jre-jammy AS java
FROM python:${PYTHON_VERSION}-slim-bookworm

COPY --from=java /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV SPARK_HOME=/opt/spark
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PYSPARK_DRIVER_PYTHON=/usr/local/bin/python
ENV PYTHONPATH=/workspace
ENV PYARROW_IGNORE_TIMEZONE=1
ENV SPARK_LOCAL_DIRS=/tmp/spark
ENV SPARK_DAEMON_MEMORY=512m
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV COVERAGE_FILE=/coverage/.coverage

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean

ARG SPARK_VERSION=3.5.1
WORKDIR /workspace
COPY pyproject.toml setup.py README.md LICENSE ./
COPY flaml ./flaml
# The Spark suite collects the existing test extra, but does not use CUDA.
RUN python -m pip install --no-cache-dir --upgrade pip wheel "setuptools<82" \
    && python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -e ".[test,forecast]" "pyspark==${SPARK_VERSION}" "pandas>=2.0,<3" \
    && ln -s "$(python -c 'import os,pyspark; print(os.path.dirname(pyspark.__file__))')" /opt/spark \
    && mkdir -p /opt/spark/conf /shared /coverage \
    && python -c "import pathlib,sysconfig; (pathlib.Path(sysconfig.get_paths()['purelib'])/'coverage_subprocess.pth').write_text('import coverage; coverage.process_startup()\n')"

COPY test ./test
COPY notebook ./notebook
COPY pytest.ini .coveragerc ./
COPY test/spark/containers/spark-defaults.conf /opt/spark/conf/spark-defaults.conf
CMD ["coverage", "run", "-m", "test.spark.containers.run_tests"]
