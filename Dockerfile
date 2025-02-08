# Use PostgreSQL 16.6 as base image
FROM postgres:16.6

# Install build dependencies as per AGE documentation
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-16 \
    curl \
    libreadline-dev \
    zlib1g-dev \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector 0.8.0
RUN curl -L -o /tmp/vector.tar.gz https://github.com/pgvector/pgvector/archive/refs/tags/v0.8.0.tar.gz \
    && cd /tmp \
    && tar -xzf vector.tar.gz \
    && cd pgvector-0.8.0 \
    && make \
    && make install

# Install pg_cron
RUN git clone https://github.com/citusdata/pg_cron.git \
    && cd pg_cron \
    && make \
    && make install

# Clone and build Apache AGE 1.5.0 using PostgreSQL build system
RUN git clone --branch PG16/v1.5.0-rc0 https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && make \
    && make install

# Add initialization script to enable extensions
RUN mkdir -p /docker-entrypoint-initdb.d
COPY init-extensions.sql /docker-entrypoint-initdb.d/

# Set environment variables
ENV POSTGRES_DB=fastrag
ENV POSTGRES_USER=fastrag
ENV POSTGRES_PASSWORD=fastrag

# Performance tuning
ENV POSTGRES_SHARED_BUFFERS=2GB
ENV POSTGRES_EFFECTIVE_CACHE_SIZE=6GB
ENV POSTGRES_MAINTENANCE_WORK_MEM=512MB
ENV POSTGRES_RANDOM_PAGE_COST=1.1
ENV POSTGRES_EFFECTIVE_IO_CONCURRENCY=200
ENV POSTGRES_WORK_MEM=16MB
ENV POSTGRES_MIN_WAL_SIZE=2GB
ENV POSTGRES_MAX_WAL_SIZE=8GB

# Enable required extensions in postgresql.conf
RUN echo "shared_preload_libraries = 'pg_cron,age,vector'" >> /usr/share/postgresql/postgresql.conf.sample
RUN echo "cron.database_name = 'fastrag'" >> /usr/share/postgresql/postgresql.conf.sample

# Create a volume for persistent data
VOLUME ["/var/lib/postgresql/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD pg_isready -U fastrag -d fastrag || exit 1

# Expose PostgreSQL port
EXPOSE 5432 