FROM python:3.12-slim

# OS deps
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Litestream
RUN curl -L https://github.com/benbjohnson/litestream/releases/download/v0.3.13/litestream-v0.3.13-linux-amd64.tar.gz \
  | tar -xz -C /usr/local/bin

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
COPY litestream.yml /etc/litestream.yml
COPY start.sh /start.sh
RUN chmod +x /start.sh

# The /data path is where SQLite will live
VOLUME ["/data"]

EXPOSE 8000
CMD ["/start.sh"]