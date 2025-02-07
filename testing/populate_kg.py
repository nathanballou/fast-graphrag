import logging
from circlemind import Circlemind
from zep_cloud.client import AsyncZep
from zep_cloud import Message
from typing import List, Dict, Any
import json
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


circlemind_client = Circlemind()


def setup_task_kg():
    """Setup knowledge graph for service tasks and pricing"""
    circlemind_client.create_graph(
        graph_id="service-tasks",
        domain="""
        Learn from full conversation threads about service tasks to answer two key questions:

        1. "What are three questions that we should ask the user to determine an accurate quote?"
           - Learn which questions were most important for accurate quotes
           - Understand what information was missing when quotes were inaccurate
           - Identify which answers most affected final pricing
           - Consider the full context of conversations including follow-up questions and clarifications
        
        2. "Analyze this service task and provide a detailed price estimate."
           - Learn what materials and quantities were needed
           - Learn how long tasks actually took (from both estimates and actual reported times)
           - Learn what conditions affected the price
           - Learn what made similar jobs cost more or less
           - Consider all details discussed throughout the conversation thread
        """,
        example_queries=[
            "What are three questions that we should ask the user to determine an accurate quote for {task}?",
            "Analyze this service task and provide a detailed price estimate: {task}"
        ],
        entity_types=[
            "Question",          # e.g., "What is the ceiling height?"
            "Material",          # e.g., "Elfa shelf unit"
            "Quantity",          # e.g., "3 units"
            "Labor",            # e.g., "Installation time"
            "Condition",        # e.g., "Third floor walkup"
            "Price",            # e.g., "$500 for materials"
            "Impact",           # e.g., "Adds 2 hours to installation"
            "Requirement",      # e.g., "Needs special mounting hardware"
            "Risk",             # e.g., "Potential wall damage"
            "Constraint"        # e.g., "Must complete in one day"
        ]
    )

def populate_graphs(slack_messages: List[Dict]):
    """Populate CircleMind with conversation data"""
    
    # Sort messages by timestamp
    sorted_messages = sorted(slack_messages, key=lambda x: x['timestamp'])
    
    # Build one continuous conversation
    context_parts = []
    
    for msg in sorted_messages:
        # Add timestamp and message
        context_parts.append(f"[{msg['timestamp']}] {msg['text']}")
    
    # Join all messages into one context
    context = '\n'.join(context_parts)
    
    logger.info("Adding conversation to CircleMind...")
    logger.debug(f"Context being added:\n{context}")
    
    try:
        response = circlemind_client.add(
            memory=context,
            graph_id="service-tasks",
            metadata={
                "start_time": sorted_messages[0]['timestamp'],
                "end_time": sorted_messages[-1]['timestamp'],
                "message_count": len(sorted_messages)
            }
        )
        logger.info(f"Successfully added conversation to CircleMind. Response: {response}")
    except Exception as e:
        logger.error(f"Error adding to CircleMind: {str(e)}")
        logger.error(f"Context length: {len(context)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    # # First, let's try to create the graph
    # try:
    #     setup_task_kg()
    #     logger.info("Successfully set up knowledge graph")
    # except Exception as e:
    #     logger.error(f"Error setting up knowledge graph: {str(e)}")
    #     logger.error(traceback.format_exc())

    try:
        # Load message data
        slack_messages = None
        
        # Debug: Check if files exist
        logger.info("Checking for input files...")
        logger.info(f"slack_history.json exists: {os.path.exists('slack_history.json')}")
        
        # Load slack messages
        if os.path.exists('slack_history.json'):
            with open('slack_history.json', 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Read {len(content)} bytes from slack_history.json")
                data = json.loads(content)
                if isinstance(data, dict):
                    slack_messages = data.get('messages', [])
                else:
                    slack_messages = data
                logger.info(f"Loaded {len(slack_messages) if slack_messages else 0} messages from slack_history.json")
                logger.info(f"Sample message: {slack_messages[0] if slack_messages else 'No messages'}")

        # Populate graphs with available data
        if slack_messages:
            logger.info("Starting to populate graphs...")
            populate_graphs(slack_messages)
            logger.info("Finished populating graphs")
        else:
            logger.error("No messages loaded from files")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()