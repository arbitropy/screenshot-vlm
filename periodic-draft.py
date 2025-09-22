import asyncio
import base64
import io
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import litellm
import numpy as np
import pyautogui
from PIL import Image
from pydantic import BaseModel, Field


class ScreenAnalysisResult(BaseModel):
    """
    Pydantic model for structured VLM response.
    You can customize this based on your needs.
    """
    reasoning: Optional[str] = FI


class VLMScreenshotAnalyzer:
    """
    A class to capture screenshots at variable intervals and analyze them using OpenAI VLM.
    """
    
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        api_key: Optional[str] = None,
        min_interval: int = 5,
        max_interval: int = 30,
        save_screenshots: bool = True,
        screenshot_dir: str = "screenshots"
    ):
        """
        Initialize the VLM Screenshot Analyzer.
        
        Args:
            model: OpenAI model to use for vision analysis
            api_key: OpenAI API key (if None, will use environment variable)
            min_interval: Minimum seconds between screenshots
            max_interval: Maximum seconds between screenshots
            save_screenshots: Whether to save screenshots to disk
            screenshot_dir: Directory to save screenshots
        """
        self.model = model
        self.api_key = api_key
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.save_screenshots = save_screenshots
        self.screenshot_dir = Path(screenshot_dir)
        
        if self.save_screenshots:
            self.screenshot_dir.mkdir(exist_ok=True)
    
    def capture_screenshot(self) -> tuple[Image.Image, str]:
        """
        Capture a screenshot and return PIL Image and base64 encoded string.
        
        Returns:
            Tuple of (PIL Image, base64 encoded string)
        """
        # Capture screenshot
        screenshot = pyautogui.screenshot()
        
        # Convert to base64 for API
        buffer = io.BytesIO()
        screenshot.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        data_url = f"data:image/png;base64,{image_base64}"
        
        # Optionally save to disk
        if self.save_screenshots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = self.screenshot_dir / f"screenshot_{timestamp}.png"
            screenshot.save(filename)
            print(f"Screenshot saved: {filename}")
        
        return screenshot, data_url
    
    async def analyze_screenshot(self, image_data_url: str, custom_prompt: Optional[str] = None) -> ScreenAnalysisResult:
        """
        Send screenshot to OpenAI VLM for analysis.
        
        Args:
            image_data_url: Base64 encoded image data URL
            custom_prompt: Custom prompt for analysis
            
        Returns:
            Structured analysis result
        """
        # Default prompt - you can customize this
        default_prompt = """
        Analyze this screenshot and provide:
        1. A brief description of what's visible
        2. Your confidence level (0-1) in the analysis
        3. List of detected UI elements or content
        4. Suggested actions a user might take
        
        Please respond in a structured format that matches the expected JSON schema.
        """
        
        prompt = custom_prompt or default_prompt
        
        try:
            # Use litellm for OpenAI VLM call
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url,
                                    "detail": "high"  # Use high detail for better analysis
                                }
                            }
                        ]
                    }
                ],
                api_key=self.api_key,
                temperature=0.3,  # Lower temperature for more consistent responses
                max_tokens=1000
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # For now, create a basic structured response
            # You can enhance this to parse JSON from the model response
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Simple parsing - you might want to improve this based on your prompt
            result = ScreenAnalysisResult(
                description=content[:200] + "..." if len(content) > 200 else content,
                confidence=0.8,  # You can extract this from the response
                detected_elements=["screen_content"],  # Parse from response
                action_suggestions=["analyze_further"],  # Parse from response
                timestamp=current_time
            )
            
            return result
            
        except Exception as e:
            print(f"Error analyzing screenshot: {e}")
            # Return a default result on error
            return ScreenAnalysisResult(
                description=f"Error analyzing screenshot: {str(e)}",
                confidence=0.0,
                detected_elements=[],
                action_suggestions=[],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
    
    def get_next_interval(self) -> int:
        """Get random interval between min and max seconds."""
        return random.randint(self.min_interval, self.max_interval)
    
    async def run_continuous_analysis(
        self, 
        duration_minutes: Optional[int] = None,
        custom_prompt: Optional[str] = None
    ):
        """
        Run continuous screenshot analysis.
        
        Args:
            duration_minutes: How long to run (None for infinite)
            custom_prompt: Custom prompt for all analyses
        """
        print(f"Starting VLM Screenshot Analyzer...")
        print(f"Model: {self.model}")
        print(f"Interval: {self.min_interval}-{self.max_interval} seconds")
        print(f"Duration: {'Infinite' if duration_minutes is None else f'{duration_minutes} minutes'}")
        
        start_time = time.time()
        screenshot_count = 0
        
        try:
            while True:
                # Check duration limit
                if duration_minutes and (time.time() - start_time) > (duration_minutes * 60):
                    print(f"Completed {duration_minutes} minutes of analysis")
                    break
                
                screenshot_count += 1
                print(f"\n--- Screenshot #{screenshot_count} ---")
                
                # Capture screenshot
                screenshot, data_url = self.capture_screenshot()
                print(f"Screenshot captured: {screenshot.size}")
                
                # Analyze with VLM
                print("Analyzing with VLM...")
                analysis = await self.analyze_screenshot(data_url, custom_prompt)
                
                # Print results
                print(f"Analysis Result:")
                print(f"  Description: {analysis.description}")
                print(f"  Confidence: {analysis.confidence}")
                print(f"  Elements: {analysis.detected_elements}")
                print(f"  Suggestions: {analysis.action_suggestions}")
                print(f"  Timestamp: {analysis.timestamp}")
                
                # Wait for next interval
                next_interval = self.get_next_interval()
                print(f"Waiting {next_interval} seconds until next screenshot...")
                await asyncio.sleep(next_interval)
                
        except KeyboardInterrupt:
            print(f"\nStopped by user after {screenshot_count} screenshots")
        except Exception as e:
            print(f"Error in continuous analysis: {e}")


async def main():
    """
    Main function to run the VLM screenshot analyzer.
    """
    # Initialize analyzer
    analyzer = VLMScreenshotAnalyzer(
        model="gpt-4-vision-preview",  # or "gpt-4o" for latest
        # api_key="your-api-key-here",  # or set OPENAI_API_KEY environment variable
        min_interval=10,  # Minimum 10 seconds between screenshots
        max_interval=60,  # Maximum 60 seconds between screenshots
        save_screenshots=True,
        screenshot_dir="screenshots"
    )
    
    # Custom prompt for your specific use case
    custom_prompt = """
    Analyze this screenshot and tell me:
    1. What application or website is currently active?
    2. What is the user likely doing?
    3. Are there any notable UI elements or content?
    4. What actions might the user take next?
    
    Be concise but detailed in your analysis.
    """
    
    # Run for 30 minutes (or remove duration_minutes for infinite)
    await analyzer.run_continuous_analysis(
        duration_minutes=30,
        custom_prompt=custom_prompt
    )


if __name__ == "__main__":
    # Run the analyzer
    asyncio.run(main())