import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPReportGenerator:
    """
    NLP Report Generator for Solana Liquidity Pool Analysis
    
    Uses Google Vertex AI (Gemini) for natural language reports and summaries
    about liquidity pool data, predictions, and market trends.
    """
    
    def __init__(self):
        """Initialize the NLP Report Generator with API credentials"""
        # Initialize Google Vertex AI Client
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("API key not found. Set GOOGLE_API_KEY environment variable.")
        else:
            try:
                self.client = genai.Client(vertexai=True, api_key=self.api_key)
                logger.info("Successfully initialized Google Vertex AI Gemini model")
            except Exception as e:
                logger.error(f"Error initializing Google Gemini model: {e}")
                self.api_key = None
            
    def has_api_key(self) -> bool:
        """Check if API key is available"""
        return bool(self.api_key)
    
    def generate_pool_summary(self, pool_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a natural language summary for a single pool
        
        Args:
            pool_data: Dictionary containing pool metrics and details
            
        Returns:
            String containing the generated summary or None if generation failed
        """
        if not self.has_api_key():
            return "API key required for summary generation. Please set GOOGLE_API_KEY environment variable."
        
        try:
            # Prepare prompt with pool data
            prompt = f"""
            You are a professional crypto analyst specializing in Solana DeFi pools.
            
            Generate a concise, informative summary of this Solana liquidity pool:
            
            Pool Name: {pool_data.get('name', 'Unknown')}
            Pool ID: {pool_data.get('pool_id', 'Unknown')}
            DEX: {pool_data.get('dex', 'Unknown')}
            Category: {pool_data.get('category', 'Unknown')}
            Current Liquidity: ${pool_data.get('liquidity', 0):,.2f}
            24h Volume: ${pool_data.get('volume_24h', 0):,.2f}
            Current APR: {pool_data.get('apr', 0):.2f}%
            TVL Change (24h): {pool_data.get('tvl_change_24h', 0):.2f}%
            APR Change (24h): {pool_data.get('apr_change_24h', 0):.2f}%
            
            Highlight key metrics, significant changes, risk level, and potential opportunities.
            Keep the summary under 150 words and make it accessible to crypto investors.
            Include a brief prediction about potential future performance.
            """
            
            # Generate response using Google Vertex AI (Gemini)
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt
            )
            
            # Extract and return text
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating pool summary: {e}")
            return None
    
    def generate_market_report(self, pools_data: List[Dict[str, Any]], 
                              top_predictions: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate a comprehensive market report based on multiple pools and predictions
        
        Args:
            pools_data: List of dictionaries containing pool metrics
            top_predictions: List of prediction results for top pools
            
        Returns:
            String containing the generated report or None if generation failed
        """
        if not self.has_api_key():
            return "API key required for report generation. Please set GOOGLE_API_KEY environment variable."
        
        try:
            # Prepare the top 5 pools by liquidity 
            top_by_liquidity = sorted(pools_data, key=lambda x: x.get('liquidity', 0), reverse=True)[:5]
            
            # Prepare the top 5 pools by APR
            top_by_apr = sorted(pools_data, key=lambda x: x.get('apr', 0), reverse=True)[:5]
            
            # Format data for the prompt
            top_liquidity_str = "\n".join([
                f"- {p.get('name', 'Unknown')}: ${p.get('liquidity', 0):,.2f}, APR: {p.get('apr', 0):.2f}%" 
                for p in top_by_liquidity
            ])
            
            top_apr_str = "\n".join([
                f"- {p.get('name', 'Unknown')}: APR: {p.get('apr', 0):.2f}%, Liquidity: ${p.get('liquidity', 0):,.2f}" 
                for p in top_by_apr
            ])
            
            # Prepare top predictions
            predictions_str = "\n".join([
                f"- {p.get('pool_name', 'Unknown')}: Predicted APR: {p.get('predicted_apr', 0):.2f}%, " +
                f"Risk Score: {p.get('risk_score', 0):.2f}"
                for p in top_predictions[:5]
            ])
            
            # Calculate market averages
            avg_apr = sum(p.get('apr', 0) for p in pools_data) / len(pools_data) if pools_data else 0
            total_liquidity = sum(p.get('liquidity', 0) for p in pools_data)
            
            # Prepare prompt for Google Vertex AI (Gemini)
            prompt = f"""
            You are a professional crypto market analyst specializing in DeFi liquidity pools on Solana.
            
            Generate a comprehensive Solana DeFi market report based on this liquidity pool data.
            
            Date: {datetime.now().strftime('%Y-%m-%d')}
            
            MARKET OVERVIEW:
            Total Tracked Liquidity: ${total_liquidity:,.2f}
            Average APR: {avg_apr:.2f}%
            Number of Pools Analyzed: {len(pools_data)}
            
            TOP POOLS BY LIQUIDITY:
            {top_liquidity_str}
            
            TOP POOLS BY APR:
            {top_apr_str}
            
            TOP PREDICTED PERFORMERS:
            {predictions_str}
            
            Your report should include:
            1. An executive summary of the current state of Solana liquidity pools
            2. Analysis of notable trends across different pool categories
            3. Identification of potential opportunities and risks
            4. Brief outlook for the coming week
            
            Write in a professional but accessible style suitable for crypto investors.
            The report should be around 400-500 words.
            """
            
            # Generate response using Google Vertex AI (Gemini)
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt
            )
            
            # Extract and return text
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating market report: {e}")
            return None
            
    def generate_specific_analysis(self, query: str, context_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a custom analysis based on a specific user query
        
        Args:
            query: User's specific question or analysis request
            context_data: Dictionary containing relevant pool data for context
            
        Returns:
            String containing the generated analysis or None if generation failed
        """
        if not self.has_api_key():
            return "API key required for analysis generation. Please set GOOGLE_API_KEY environment variable."
        
        try:
            # Format context data for the prompt
            context_str = json.dumps(context_data, indent=2)
            
            # Prepare prompt for Google Vertex AI (Gemini)
            prompt = f"""
            You are an expert crypto analyst specializing in detailed analysis of Solana DeFi pools.
            
            Analyze the following Solana liquidity pool data to answer this specific question:
            
            USER QUERY: {query}
            
            CONTEXT DATA:
            {context_str}
            
            Provide a detailed, data-driven analysis that directly addresses the query.
            Support your analysis with specific metrics from the provided data.
            Make your response accessible to crypto investors while maintaining technical accuracy.
            """
            
            # Generate response using Google Vertex AI (Gemini)
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=prompt
            )
            
            # Extract and return text
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating specific analysis: {e}")
            return None
    
    def save_report(self, report: str, report_type: str) -> str:
        """
        Save a generated report to a file
        
        Args:
            report: The text content of the report
            report_type: Type of report (e.g., 'market', 'pool_summary')
            
        Returns:
            Path to the saved report file
        """
        try:
            # Create reports directory if it doesn't exist
            os.makedirs('reports', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"reports/{report_type}_{timestamp}.txt"
            
            # Save report to file
            with open(filename, 'w') as f:
                f.write(report)
                
            logger.info(f"Report saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return ""