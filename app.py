import os
os.environ['DISPLAY'] = ':0'

import asyncio
import base64
import io
import time
import threading
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

import gradio as gr
import litellm
import pyautogui
import pygame
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
import tkinter as tk
from tkinter import messagebox

class ScreenAnalysisResult(BaseModel):
    reasoning: str = Field(description="Model's reasoning about the screenshot's eligibility to invoke the tool")
    decision: bool = Field(description="Whether to invoke the tool or not")

class ContentBlocker:
    """Generic content blocker using VLM screenshot analysis."""
    
    def __init__(self):
        self.running = False
        self.blocking_active = False
        self.analysis_thread = None
        self.block_thread = None
        self.screenshot_dir = Path("screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
        
        # Initialize pygame for screen blocking
        pygame.init()
        
    async def capture_and_analyze(
        self,
        base_url: str,
        api_key: str,
        model: str,
        custom_prompt: str,
        save_screenshots: bool = True
    ) -> tuple[bool, str, Optional[str]]:
        """Capture screenshot and analyze with VLM."""
        try:
            # Capture screenshot
            screenshot = pyautogui.screenshot()
            
            # Convert to base64
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_url = f"data:image/png;base64,{image_base64}"
            
            # Analyze with VLM
            response = await litellm.acompletion(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
                    ]
                }],
                base_url=base_url,
                api_key=api_key,
                temperature=0.01,
                max_tokens=500,
                response_format=ScreenAnalysisResult
            )
            
            result = response.choices[0].message.content
            
            # Save screenshot with analysis if requested
            screenshot_path = None
            if save_screenshots:
                screenshot_path = self._save_screenshot_with_analysis(screenshot, result)
            
            return result.decision, result.reasoning, screenshot_path
            
        except Exception as e:
            return False, f"Error in analysis: {str(e)}", None
    
    def _save_screenshot_with_analysis(self, screenshot: Image.Image, result: ScreenAnalysisResult) -> str:
        """Save screenshot with analysis overlay."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a larger image to include analysis text
        img_width, img_height = screenshot.size
        text_height = 200
        combined_img = Image.new('RGB', (img_width, img_height + text_height), 'white')
        
        # Paste original screenshot
        combined_img.paste(screenshot, (0, 0))
        
        # Add analysis text
        draw = ImageDraw.Draw(combined_img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Format text
        decision_text = f"DECISION: {'BLOCK' if result.decision else 'ALLOW'}"
        reasoning_text = f"REASONING: {result.reasoning[:150]}..."
        
        # Draw text with background
        text_y = img_height + 10
        draw.rectangle([(0, img_height), (img_width, img_height + text_height)], fill='lightgray')
        draw.text((10, text_y), decision_text, fill='red' if result.decision else 'green', font=font)
        draw.text((10, text_y + 30), reasoning_text, fill='black', font=font)
        
        # Save
        filename = self.screenshot_dir / f"analysis_{timestamp}.png"
        combined_img.save(filename)
        return str(filename)
    
    def block_screen(self, block_image_path: Optional[str], duration: int):
        """Block screen with image and countdown."""
        if self.blocking_active:
            return
            
        self.blocking_active = True
        
        # Get screen dimensions
        screen_info = pygame.display.Info()
        screen_width, screen_height = screen_info.current_w, screen_info.current_h
        
        # Create fullscreen window
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption("Content Blocked")
        
        # Load block image if provided
        block_image = None
        if block_image_path and Path(block_image_path).exists():
            try:
                block_image = pygame.image.load(block_image_path)
                block_image = pygame.transform.scale(block_image, (screen_width, screen_height))
            except:
                block_image = None
        
        # Block input (disable keyboard and mouse)
        self._block_input()
        
        font = pygame.font.Font(None, 72)
        clock = pygame.time.Clock()
        
        start_time = time.time()
        
        while time.time() - start_time < duration and self.blocking_active:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    # Allow escape key to exit (for safety)
                    self.blocking_active = False
                    break
            
            # Clear screen
            if block_image:
                screen.blit(block_image, (0, 0))
            else:
                screen.fill((255, 0, 0))  # Red background
            
            # Draw countdown
            remaining = int(duration - (time.time() - start_time))
            countdown_text = font.render(f"Blocked - {remaining}s remaining", True, (255, 255, 255))
            text_rect = countdown_text.get_rect(center=(screen_width//2, screen_height//2))
            screen.blit(countdown_text, text_rect)
            
            # Draw message
            message_font = pygame.font.Font(None, 36)
            message_text = message_font.render("Inappropriate content detected. Please wait...", True, (255, 255, 255))
            message_rect = message_text.get_rect(center=(screen_width//2, screen_height//2 + 100))
            screen.blit(message_text, message_rect)
            
            pygame.display.flip()
            clock.tick(60)
        
        # Cleanup
        self._unblock_input()
        pygame.quit()
        pygame.init()  # Reinitialize for next use
        self.blocking_active = False
    
    def _block_input(self):
        """Block keyboard and mouse input (platform-specific implementation needed)."""
        # This is a placeholder - actual implementation would require platform-specific code
        # For Linux: can use xinput to disable devices
        # For Windows: can use Windows API to block input
        # For macOS: can use Quartz Event Services
        pass
    
    def _unblock_input(self):
        """Unblock keyboard and mouse input."""
        # Corresponding unblock implementation
        pass
    
    async def run_monitoring(
        self,
        base_url: str,
        api_key: str,
        model: str,
        period: int,
        custom_prompt: str,
        save_screenshots: bool,
        block_image_path: Optional[str],
        block_duration: int,
        status_callback: Optional[Callable] = None
    ):
        """Main monitoring loop."""
        self.running = True
        
        while self.running:
            try:
                if status_callback:
                    status_callback("🔍 Analyzing screenshot...")
                
                should_block, reasoning, screenshot_path = await self.capture_and_analyze(
                    base_url, api_key, model, custom_prompt, save_screenshots
                )
                
                if status_callback:
                    status_callback(f"📊 Decision: {'BLOCK' if should_block else 'ALLOW'}\n{reasoning}")
                
                if should_block:
                    if status_callback:
                        status_callback(f"🚫 Blocking screen for {block_duration} seconds...")
                    
                    # Run blocking in separate thread to not block the monitoring loop
                    self.block_thread = threading.Thread(
                        target=self.block_screen,
                        args=(block_image_path, block_duration)
                    )
                    self.block_thread.start()
                    self.block_thread.join()  # Wait for blocking to complete
                
                if self.running:  # Check if still running after potential blocking
                    await asyncio.sleep(period)
                    
            except Exception as e:
                if status_callback:
                    status_callback(f"❌ Error: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def start_monitoring(self, **kwargs):
        """Start monitoring in background thread."""
        if self.running:
            return "Already running!"
        
        self.analysis_thread = threading.Thread(
            target=lambda: asyncio.run(self.run_monitoring(**kwargs))
        )
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        return "✅ Monitoring started!"
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        self.blocking_active = False
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2)
        
        return "🛑 Monitoring stopped!"

# Global blocker instance
blocker = ContentBlocker()

# Gradio Interface
def create_interface():
    with gr.Blocks(
        title="AI Content Blocker",
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="orange")
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🛡️ AI Content Blocker</h1>
            <p>Intelligent content filtering using Vision Language Models</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## ⚙️ Configuration")
                
                with gr.Group():
                    base_url = gr.Textbox(
                        label="🌐 Base URL",
                        placeholder="https://api.openai.com/v1",
                        value="https://ollama-scm.transformsai.org/v1/"
                    )
                    
                    api_key = gr.Textbox(
                        label="🔑 API Key",
                        placeholder="Enter your API key",
                        type="password",
                        value="kobold"
                    )
                    
                    model = gr.Textbox(
                        label="🤖 Model Name",
                        placeholder="gpt-4-vision-preview",
                        value="openai/gpt-4o-mini"
                    )
                
                with gr.Group():
                    period = gr.Slider(
                        label="⏱️ Screenshot Period (seconds)",
                        minimum=1,
                        maximum=60,
                        value=5,
                        step=1
                    )
                    
                    block_duration = gr.Slider(
                        label="🚫 Block Duration (seconds)",
                        minimum=5,
                        maximum=300,
                        value=30,
                        step=5
                    )
                
                save_screenshots = gr.Checkbox(
                    label="💾 Save Screenshots with Analysis",
                    value=True
                )
                
                block_image = gr.File(
                    label="🖼️ Block Screen Image (optional)",
                    file_types=["image"]
                )
                
                custom_prompt = gr.Textbox(
                    label="📝 Analysis Prompt",
                    placeholder="Describe what content should be blocked...",
                    lines=5,
                    value="Analyze the screenshot and determine if it contains inappropriate content (politics, adult content, violent content, or other distracting material). Provide reasoning and a boolean decision on whether the blocking tool should be invoked."
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## 🎮 Control Panel")
                
                with gr.Group():
                    start_btn = gr.Button(
                        "▶️ Start Monitoring",
                        variant="primary",
                        size="lg"
                    )
                    
                    stop_btn = gr.Button(
                        "⏹️ Stop Monitoring",
                        variant="stop",
                        size="lg"
                    )
                
                status_display = gr.Textbox(
                    label="📊 Status",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                    value="Ready to start monitoring..."
                )
        
        # Event handlers
        def start_monitoring(*args):
            kwargs = {
                'base_url': args[0],
                'api_key': args[1],
                'model': args[2],
                'period': int(args[3]),
                'block_duration': int(args[4]),
                'save_screenshots': args[5],
                'block_image_path': args[6].name if args[6] else None,
                'custom_prompt': args[7],
                'status_callback': lambda msg: status_display.update(value=msg)
            }
            return blocker.start_monitoring(**kwargs)
        
        start_btn.click(
            fn=start_monitoring,
            inputs=[
                base_url, api_key, model, period, block_duration,
                save_screenshots, block_image, custom_prompt
            ],
            outputs=None
        )
        
        stop_btn.click(
            fn=blocker.stop_monitoring,
            inputs=None,
            outputs=None
        )
        
        # Auto-refresh status (placeholder - would need WebSocket for real-time updates)
        gr.HTML("""
        <div style="margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
            <p><strong>⚠️ Safety Notes:</strong></p>
            <ul>
                <li>Press ESC during blocking to emergency exit</li>
                <li>Screenshots are saved locally in the 'screenshots' folder</li>
                <li>Ensure your VLM model supports vision capabilities</li>
                <li>Test with a short block duration first</li>
            </ul>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )