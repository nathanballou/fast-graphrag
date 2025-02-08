-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_cron;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Create schema
CREATE SCHEMA IF NOT EXISTS fastrag;

-- Create a graph in AGE
SELECT create_graph('fastrag');

-- Create tables for storing embeddings and metadata
CREATE TABLE IF NOT EXISTS fastrag.vectors (
    id SERIAL PRIMARY KEY,
    namespace TEXT NOT NULL,
    external_id TEXT NOT NULL,
    embedding vector(768),  -- Adjust vector dimension based on your model
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(namespace, external_id)
);

-- Create tables for key-value storage
CREATE TABLE IF NOT EXISTS fastrag.key_values (
    id SERIAL PRIMARY KEY,
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    idx INTEGER NOT NULL GENERATED ALWAYS AS IDENTITY,
    value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(namespace, key)
);

-- Create table for blob storage
CREATE TABLE IF NOT EXISTS fastrag.blobs (
    id SERIAL PRIMARY KEY,
    namespace TEXT NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_vectors_namespace_id ON fastrag.vectors(namespace, external_id);
CREATE INDEX IF NOT EXISTS idx_vectors_embedding ON fastrag.vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_vectors_metadata ON fastrag.vectors USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_key_values_namespace_key ON fastrag.key_values(namespace, key);
CREATE INDEX IF NOT EXISTS idx_key_values_namespace_idx ON fastrag.key_values(namespace, idx);

CREATE INDEX IF NOT EXISTS idx_blobs_namespace_created ON fastrag.blobs(namespace, created_at DESC);

-- Set up permissions
GRANT USAGE ON SCHEMA fastrag TO fastrag;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA fastrag TO fastrag;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA fastrag TO fastrag;

-- Create function to automatically clean expired cache entries
CREATE OR REPLACE FUNCTION fastrag.clean_expired_cache() RETURNS void AS $$
BEGIN
    DELETE FROM fastrag.cache WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job to clean expired cache (runs every hour)
SELECT cron.schedule('0 * * * *', $$SELECT fastrag.clean_expired_cache()$$); 