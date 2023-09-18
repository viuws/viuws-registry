FROM python:3

WORKDIR /usr/src/app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir click imageio pandas scikit-image universal_pathlib && \
    pip install --no-cache-dir adlfs gcsfs kerchunk ossfs s3fs sshfs swiftspec

COPY app.py ./

ENTRYPOINT ["python", "./app.py"]
