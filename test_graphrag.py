"""Test GraphRAG functionality."""

import logging
import os
import shutil

from dotenv import load_dotenv

from fast_graphrag import GraphRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."

EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]

ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

def test_file_storage():
    """Test GraphRAG with file storage."""
    try:
        logger.info("Starting file storage test...")
        
        # Create test workspace directory with absolute path
        test_workspace = os.path.abspath("./test_workspace")
        if os.path.exists(test_workspace):
            shutil.rmtree(test_workspace)
        os.makedirs(test_workspace)
        
        # Initialize GraphRAG with file storage
        rag = GraphRAG(
            working_dir=test_workspace,
            domain=DOMAIN,
            example_queries="\n".join(EXAMPLE_QUERIES),
            entity_types=ENTITY_TYPES,
            storage_type="file"
        )
        
        # Test basic operations
        test_content = "Ebenezer Scrooge was a miserly old man who lived in Victorian London. On Christmas Eve, he was visited by three spirits who showed him visions of his past, present, and future. Through these encounters, Scrooge transformed from a cold-hearted businessman into a generous and kind person. His clerk, Bob Cratchit, and Cratchit's young son Tiny Tim played important roles in his transformation."
        
        logger.info("Testing document insertion...")
        num_entities, num_relations, num_chunks = rag.insert(content=test_content)
        logger.info(f"Inserted {num_entities} entities, {num_relations} relations, and {num_chunks} chunks")
        
        # Test query
        logger.info("Testing query functionality...")
        response = rag.query(query="Who is Tiny Tim?")
        logger.info(f"Query response: {response.response}")
        
        logger.info("File storage test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"File storage test failed: {str(e)}")
        logger.error("Traceback:")
        import traceback
        traceback.print_exc()
        return False

def test_postgres_storage():
    """Test GraphRAG with PostgreSQL storage."""
    try:
        logger.info("Starting PostgreSQL storage test...")
        
        # Create test workspace directory
        test_workspace = "./test_workspace_pg"
        if os.path.exists(test_workspace):
            shutil.rmtree(test_workspace)
        os.makedirs(test_workspace)
        
        # Initialize GraphRAG with PostgreSQL storage
        rag = GraphRAG(
            working_dir=test_workspace,
            domain=DOMAIN,
            example_queries="\n".join(EXAMPLE_QUERIES),
            entity_types=ENTITY_TYPES,
            storage_type="postgres"
        )

        # Test basic operations
        test_content = "Ebenezer Scrooge was a miserly old man who lived in Victorian London. On Christmas Eve, he was visited by three spirits who showed him visions of his past, present, and future. Through these encounters, Scrooge transformed from a cold-hearted businessman into a generous and kind person. His clerk, Bob Cratchit, and Cratchit's young son Tiny Tim played important roles in his transformation."
        
        logger.info("Testing document insertion...")
        num_entities, num_relations, num_chunks = rag.insert(content=test_content)
        logger.info(f"Inserted {num_entities} entities, {num_relations} relations, and {num_chunks} chunks")
        
        # Test query
        logger.info("Testing query functionality...")
        response = rag.query(query="Who is Tiny Tim?")
        logger.info(f"Query response: {response.response}")
        
        logger.info("PostgreSQL storage test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"PostgreSQL storage test failed: {str(e)}")
        logger.error("Traceback:")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("Starting GraphRAG tests...")
    
    # Run file storage test
    file_success = test_file_storage()
    
    # Run PostgreSQL storage test if environment variables are set
    if os.getenv("PG_HOST"):
        pg_success = test_postgres_storage()
    else:
        logger.info("Skipping PostgreSQL test (PG_HOST not set)")
        pg_success = True
    
    success = file_success and pg_success
    logger.info(f"All tests completed. Success: {success}")
    return success

if __name__ == "__main__":
    main()
