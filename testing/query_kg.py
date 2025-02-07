"""Script for querying knowledge graphs to enhance LLM pricing decisions"""

from circlemind import Circlemind
from google import genai
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import logging

load_dotenv()

# Initialize clients
gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
circlemind_client = Circlemind(api_key=os.getenv('CIRCLEMIND_API_KEY'))

logger = logging.getLogger(__name__)

class LineItem(BaseModel):
    """Individual line item for materials or labor"""
    item: str = Field(description="Description of the item")
    quantity: float = Field(description="Quantity needed")
    unit_price: float = Field(description="Price per unit in USD")
    total: float = Field(description="Total price for this line item")

class MaterialCost(BaseModel):
    """Cost details for materials"""
    items: List[LineItem] = Field(description="List of material line items")
    total: float = Field(description="Total material cost")

class LaborCost(BaseModel):
    """Labor cost breakdown"""
    items: List[LineItem] = Field(description="List of labor line items")
    total: float = Field(description="Total labor cost")

class PriceEstimate(BaseModel):
    """Complete price estimate for a service task"""
    materials: MaterialCost = Field(description="Materials breakdown")
    labor: LaborCost = Field(description="Labor breakdown")
    total: float = Field(description="Total estimated cost")
    reasoning: str = Field(description="Brief explanation of key factors")

class PropertyLocation(BaseModel):
    """Represents a specific location within a property"""
    room: str = Field(description="Room or area name")
    floor: str = Field(description="Floor level")
    issues: List[Dict[str, Any]] = Field(description="Known issues with timestamps and status")

class PropertyProfile(BaseModel):
    """Complete property profile"""
    locations: List[PropertyLocation] = Field(description="Different locations in the property")
    square_footage: float = Field(description="Total square footage")
    year_built: int = Field(description="Year the property was built")
    last_renovation: int = Field(description="Year of last major renovation")

def get_kg_context(task_description: str) -> Dict[str, str]:
    """Get relevant context from knowledge graphs"""
    try:
        task_result = circlemind_client.query(
            query=f"""
            Find pricing details and requirements for:
            '{task_description}'
            
            Include:
            - Typical costs
            - Required materials
            - Labor estimates
            """,
            graph_id="service-tasks"
        )
        
        print("\n=== CircleMind Task Context ===")
        print(f"Query: {task_description}")
        print(f"Response: {task_result.response}")
        
        return {
            "task_context": task_result.response
        }
    except Exception as e:
        logger.error(f"Error getting KG context: {e}")
        return {"task_context": "No context available due to error"}

def get_price_estimate(task_description: str, with_context: bool = True) -> Dict[str, Any]:
    """Get structured price estimate from Gemini, optionally enhanced with KG context"""
    try:
        prompt = f"""Analyze this service task and provide a detailed price estimate.
        
        Task: {task_description}

        Please provide:
        - Description of required materials and their total cost
        - Description of labor requirements and total cost
        - Total price estimate
        - Reasoning behind the estimate
        """
        
        if with_context:
            kg_context = get_kg_context(task_description)
            prompt += f"""
            Consider this additional context:
            
            Similar Past Tasks:
            {kg_context['task_context']}
            """

        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': PriceEstimate,
            }
        )
        
        return {
            "prompt_used": prompt,
            "response": response.parsed,
            "with_context": with_context
        }
    except Exception as e:
        logger.error(f"Error getting price estimate: {e}")
        raise

def get_key_questions(task_description: str) -> List[str]:
    """Get important questions to ask before quoting based on task context"""
    try:
        task_result = circlemind_client.query(
            query=f"""
            Find common considerations and requirements for:
            '{task_description}'
            """,
            graph_id="service-tasks"
        )
        
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"""
            Based on this context about {task_description}:
            {task_result.response}
            
            What are the 3 most important questions to ask the customer before providing a quote?
            Return only the numbered questions, no other text.
            """,
        )
        
        return response.text.strip().split('\n')
    except Exception as e:
        logger.error(f"Error getting key questions: {e}")
        return ["Error getting questions"]

def compare_estimates():
    """Compare structured price estimates with and without KG context"""
    try:
        task_description = "Install a new closet system"
        print(f"\nPrice Comparison: {task_description}")
        print("-" * 50)
        
        # Get estimates with and without context
        without_context = get_price_estimate(task_description, with_context=False)
        with_context = get_price_estimate(task_description, with_context=True)
        
        def print_estimate(estimate, title):
            print(f"\n{title}\n")
            print("Materials:")
            for item in estimate['response'].materials.items:
                print(f"  {item.item:<30} {item.quantity:>4.1f} x ${item.unit_price:>6.2f} = ${item.total:>8.2f}")
            print(f"  {'Material Subtotal':>42} = ${estimate['response'].materials.total:>8.2f}")
            
            print("\nLabor:")
            for item in estimate['response'].labor.items:
                print(f"  {item.item:<30} {item.quantity:>4.1f} x ${item.unit_price:>6.2f} = ${item.total:>8.2f}")
            print(f"  {'Labor Subtotal':>42} = ${estimate['response'].labor.total:>8.2f}")
            
            print("\nTOTAL ESTIMATE:".ljust(44) + f"${estimate['response'].total:>8.2f}")
            print(f"\nKey Factors: {estimate['response'].reasoning}")
            print("-" * 50)
        
        # Suppress CircleMind context printing
        logging.getLogger().setLevel(logging.WARNING)
        
        print_estimate(without_context, "ESTIMATE WITHOUT HISTORICAL CONTEXT")
        print_estimate(with_context, "ESTIMATE WITH HISTORICAL CONTEXT")
        
    except Exception as e:
        logger.error(f"Error in compare_estimates: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    try:
        compare_estimates()
    except Exception as e:
        print(f"An error occurred: {e}") 