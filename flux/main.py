import os
import uuid
import requests
import json
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
from supabase import create_client, Client
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Historical Media Generator", version="2.0.0")

# Create necessary directories
Path("outputs").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# API Keys
VEO_API_KEY = os.getenv("VEO_API_KEY")
FLUX_API_KEY = os.getenv("FLUX_API_KEY")
API_BASE_URL = "https://api.kie.ai"

# Store task status in memory
task_status_store: Dict[str, Dict[str, Any]] = {}
flux_task_store: Dict[str, Dict[str, Any]] = {}

# Historical events data
HISTORICAL_EVENTS = [
    {
        "id": "ramayana",
        "title": "Ramayana - The Great Epic",
        "activities": [
            {
                "name": "Return to Ayodhya",
                "prompt": "A vlogger walks through the gates of an ancient city ruins representing Ayodhya, with historical city gates and structures. The vlogger walks the path showing the triumphal return route. Citizens depicted in ancient carvings line the path. Golden sunset creates warm celebration atmosphere. The vlogger walks joyfully explaining the homecoming. Camera follows the vlogger's journey through the historical gateway. Vertical mobile format."
            },
        ]
    },
    {
        "id": "mahabharata",
        "title": "Mahabharata - The Greatest Battle",
        "activities": [
            {
                "name": "Abhimanyu in the Chakra",
                "prompt": "A vlogger stands at a specific Chakra Vyuha (circular formation) monument site. Stone formations arranged in historical battle pattern. The vlogger walks the circular path explaining the trap. Late afternoon sunlight creates long shadows showing the formation. Ancient stone markers indicate different positions. The vlogger explains Abhimanyu's heroic entry and tragic end. Camera shows vlogger navigating the ancient formation. Vertical mobile format."
            },
        ]
    },
]

# Response Models
class VideoGenerationResponse(BaseModel):
    task_id: str
    status: str
    video_url: Optional[str] = None
    download_url: Optional[str] = None
    message: str

class ImageGenerationResponse(BaseModel):
    task_id: str
    status: str
    image_url: Optional[str] = None
    download_url: Optional[str] = None
    message: str

class ImageGenerationRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "16:9"
    model: str = "flux-kontext-pro"
    output_format: str = "jpeg"

# Helper Functions
def get_event_by_id(event_id: str):
    """Find event by ID"""
    for event in HISTORICAL_EVENTS:
        if event["id"] == event_id:
            return event
    return None

async def upload_image_to_supabase(image_file: UploadFile) -> str:
    """Upload image to Supabase Storage and return public URL"""
    try:
        file_extension = image_file.filename.split('.')[-1] if image_file.filename else 'jpg'
        filename = f"{uuid.uuid4()}.{file_extension}"
        
        content = await image_file.read()
        
        bucket_name = "images"
        response = supabase.storage.from_(bucket_name).upload(
            file=content,
            path=filename,
            file_options={"content-type": image_file.content_type}
        )
        
        public_url = supabase.storage.from_(bucket_name).get_public_url(filename)
        logger.info(f"Image uploaded to Supabase: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"Supabase upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

# ==================== FLUX API FUNCTIONS ====================

def call_flux_api(prompt: str, input_image: Optional[str] = None, **kwargs) -> dict:
    """Call Flux API to generate or edit images"""
    headers = {
        "Authorization": f"Bearer {FLUX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "aspectRatio": kwargs.get("aspect_ratio", "16:9"),
        "model": kwargs.get("model", "flux-kontext-pro"),
        "outputFormat": kwargs.get("output_format", "jpeg"),
        "enableTranslation": kwargs.get("enable_translation", True),
        "promptUpsampling": kwargs.get("prompt_upsampling", False),
        "safetyTolerance": kwargs.get("safety_tolerance", 2)
    }
    
    # Add input image for editing mode
    if input_image:
        payload["inputImage"] = input_image
    
    # Add optional parameters
    if kwargs.get("watermark"):
        payload["watermark"] = kwargs["watermark"]
    
    if kwargs.get("callback_url"):
        payload["callBackUrl"] = kwargs["callback_url"]
    
    logger.info(f"Calling Flux API with payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/flux/kontext/generate",
            headers=headers,
            json=payload,
            timeout=30
        )
        logger.info(f"Flux API Response status: {response.status_code}")
        logger.info(f"Flux API Response content: {response.text}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Flux API request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Flux image generation API error: {e}")

def check_flux_task_status(task_id: str) -> Dict[str, Any]:
    """Check Flux task status using the record-info endpoint"""
    headers = {
        "Authorization": f"Bearer {FLUX_API_KEY}"
    }
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/flux/kontext/record-info",
            headers=headers,
            params={"taskId": task_id},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logger.error(f"Flux status check failed: {response.status_code} - {response.text}")
            return {"code": response.status_code, "msg": "Failed to check status"}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Flux status check request failed: {e}")
        return {"code": 500, "msg": str(e)}

def download_image(image_url: str, output_path: str) -> bool:
    """Download generated image to images folder"""
    try:
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Image downloaded to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Image download failed: {e}")
        return False

async def monitor_flux_generation(task_id: str):
    """Background task to monitor Flux image generation and update status"""
    max_attempts = 30  # Check for 5 minutes (30 * 10 seconds)
    attempt = 0
    
    # Initialize task status
    flux_task_store[task_id] = {
        "status": "processing",
        "image_url": None,
        "download_url": None,
        "message": "Image generation in progress"
    }
    
    while attempt < max_attempts:
        try:
            logger.info(f"Checking Flux image status for task {task_id}, attempt {attempt + 1}")
            status_data = check_flux_task_status(task_id)
            
            if status_data.get("code") == 200:
                data = status_data.get("data", {})
                success_flag = data.get("successFlag", 0)
                
                if success_flag == 1:  # Success
                    response_data = data.get("response", {})
                    image_url = response_data.get("resultImageUrl")
                    
                    if image_url:
                        output_filename = f"{task_id}.jpg"
                        output_path = f"images/{output_filename}"
                        
                        # Download the image
                        if download_image(image_url, output_path):
                            flux_task_store[task_id] = {
                                "status": "completed",
                                "image_url": image_url,
                                "download_url": f"/download-image/{task_id}",
                                "message": "Image generation completed successfully"
                            }
                            logger.info(f"Flux image generation completed for task {task_id}")
                            break
                        else:
                            flux_task_store[task_id] = {
                                "status": "completed",
                                "image_url": image_url,
                                "download_url": None,
                                "message": "Image generated but download failed"
                            }
                            logger.warning(f"Image generated but download failed for task {task_id}")
                            break
                    else:
                        flux_task_store[task_id] = {
                            "status": "error",
                            "image_url": None,
                            "download_url": None,
                            "message": "No image URL found in response"
                        }
                        logger.error(f"No image URL found for task {task_id}")
                        break
                        
                elif success_flag == 2:  # Create task failed
                    error_message = data.get("errorMessage", "Unknown error")
                    flux_task_store[task_id] = {
                        "status": "failed",
                        "image_url": None,
                        "download_url": None,
                        "message": f"Image generation failed: {error_message}"
                    }
                    logger.error(f"Flux image generation failed for task {task_id}: {error_message}")
                    break
                    
                elif success_flag == 3:  # Generation failed
                    error_message = data.get("errorMessage", "Generation failed")
                    flux_task_store[task_id] = {
                        "status": "failed",
                        "image_url": None,
                        "download_url": None,
                        "message": f"Image generation failed: {error_message}"
                    }
                    logger.error(f"Flux image generation failed for task {task_id}: {error_message}")
                    break
                    
                else:  # Still processing (success_flag == 0)
                    flux_task_store[task_id] = {
                        "status": "processing",
                        "image_url": None,
                        "download_url": None,
                        "message": f"Image generation in progress... ({attempt + 1}/{max_attempts})"
                    }
            else:
                error_msg = status_data.get("msg", "Unknown error")
                flux_task_store[task_id] = {
                    "status": "error",
                    "image_url": None,
                    "download_url": None,
                    "message": f"Status check error: {error_msg}"
                }
                logger.error(f"Flux status check error for task {task_id}: {error_msg}")
                break
                
        except Exception as e:
            logger.error(f"Error monitoring Flux task {task_id}: {e}")
            flux_task_store[task_id] = {
                "status": "error",
                "image_url": None,
                "download_url": None,
                "message": f"Monitoring error: {str(e)}"
            }
        
        # Wait before next check
        await asyncio.sleep(10)
        attempt += 1
    
    # If max attempts reached and still processing
    if attempt >= max_attempts and flux_task_store.get(task_id, {}).get("status") == "processing":
        flux_task_store[task_id] = {
            "status": "timeout",
            "image_url": None,
            "download_url": None,
            "message": "Image generation timeout after 5 minutes"
        }
        logger.error(f"Flux image generation timeout for task {task_id}")

# ==================== FLUX API ENDPOINTS ====================

@app.post("/generate-image", response_model=ImageGenerationResponse)
async def generate_image(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    aspect_ratio: str = Form("16:9"),
    model: str = Form("flux-kontext-pro"),
    output_format: str = Form("jpeg"),
    input_image: Optional[UploadFile] = File(None)
):
    """
    Generate or edit image using Flux API
    """
    input_image_url = None
    
    try:
        # Upload input image if provided (for editing)
        if input_image:
            if not input_image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Input file must be an image")
            logger.info("Uploading input image to Supabase...")
            input_image_url = await upload_image_to_supabase(input_image)
        
        # Call Flux API
        logger.info("Calling Flux API...")
        api_response = call_flux_api(
            prompt=prompt,
            input_image=input_image_url,
            aspect_ratio=aspect_ratio,
            model=model,
            output_format=output_format,
            enable_translation=True,
            prompt_upsampling=False,
            safety_tolerance=2
        )
        
        if api_response.get("code") != 200:
            raise HTTPException(
                status_code=500, 
                detail=f"Flux API Error: {api_response.get('msg', 'Unknown error')}"
            )
        
        task_id = api_response["data"]["taskId"]
        
        # Start background task to monitor image generation
        background_tasks.add_task(monitor_flux_generation, task_id)
        
        return ImageGenerationResponse(
            task_id=task_id,
            status="processing",
            message="Image generation started successfully. Use the status endpoint to check progress."
        )
        
    except Exception as e:
        logger.error(f"Flux image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-historical-image", response_model=ImageGenerationResponse)
async def generate_historical_image(
    background_tasks: BackgroundTasks,
    event_id: str = Form(...),
    activity_index: int = Form(0),
    user_image: UploadFile = File(...),
    aspect_ratio: str = Form("16:9")
):
    """
    Generate historical image with user's face using Flux
    """
    # Validate image file
    if not user_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Find the event
    event = get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Validate activity index
    if activity_index >= len(event["activities"]):
        raise HTTPException(status_code=400, detail="Invalid activity index")
    
    activity = event["activities"][activity_index]
    
    try:
        # Upload user image to Supabase
        logger.info("Uploading user image to Supabase...")
        user_image_url = await upload_image_to_supabase(user_image)
        
        # Modify prompt to include user in historical context
        flux_prompt = f"{activity['prompt']} The vlogger has the face of the person in the reference image."
        
        # Call Flux API for image editing
        logger.info("Calling Flux API for historical image...")
        api_response = call_flux_api(
            prompt=flux_prompt,
            input_image=user_image_url,
            aspect_ratio=aspect_ratio,
            model="flux-kontext-max",  # Use max model for better quality
            output_format="jpeg",
            enable_translation=True,
            prompt_upsampling=True,  # Enhance the prompt
            safety_tolerance=4,  # More permissive for historical content
            watermark=f"Historical {event['title']}"
        )
        
        if api_response.get("code") != 200:
            raise HTTPException(
                status_code=500, 
                detail=f"Flux API Error: {api_response.get('msg', 'Unknown error')}"
            )
        
        task_id = api_response["data"]["taskId"]
        
        # Start background task to monitor image generation
        background_tasks.add_task(monitor_flux_generation, task_id)
        
        return ImageGenerationResponse(
            task_id=task_id,
            status="processing",
            message=f"Historical image generation started for {event['title']}. Use the status endpoint to check progress."
        )
        
    except Exception as e:
        logger.error(f"Historical image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image-status/{task_id}")
async def get_image_status(task_id: str):
    """Check Flux image generation status and get image link"""
    task_status = flux_task_store.get(task_id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    return ImageGenerationResponse(
        task_id=task_id,
        status=task_status["status"],
        image_url=task_status.get("image_url"),
        download_url=task_status.get("download_url"),
        message=task_status["message"]
    )

@app.get("/download-image/{task_id}")
async def download_generated_image(task_id: str):
    """Download generated image file"""
    image_path = f"images/{task_id}.jpg"
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found or still processing")
    
    return FileResponse(
        image_path,
        media_type='image/jpeg',
        filename=f"generated_image_{task_id}.jpg"
    )

@app.get("/images")
async def list_generated_images():
    """List all generated images"""
    images = []
    for filename in os.listdir("images"):
        if filename.endswith(".jpg"):
            file_path = f"images/{filename}"
            stat = os.stat(file_path)
            task_id = filename.replace(".jpg", "")
            images.append({
                "filename": filename,
                "task_id": task_id,
                "size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "download_url": f"/download-image/{task_id}",
                "status": flux_task_store.get(task_id, {}).get("status", "unknown")
            })
    
    return {"images": images}

@app.get("/flux-task/{task_id}")
async def get_flux_task_details(task_id: str):
    """Get detailed Flux task information"""
    status_data = check_flux_task_status(task_id)
    return status_data

# ==================== VEO API ENDPOINTS (Existing) ====================

# ... (Keep all your existing Veo API endpoints as they are)
# The Veo functions (call_veo_api, check_veo_task_status, download_video, monitor_video_generation)
# and endpoints (/generate-video, /video-status, /download-video, /videos, /task) remain the same

@app.get("/")
async def root():
    return {
        "message": "Historical Media Generator API", 
        "version": "2.0.0",
        "features": ["Veo Video Generation", "Flux Image Generation"]
    }

@app.get("/events")
async def get_events():
    """Get all available historical events"""
    return {
        "events": [
            {
                "id": event["id"],
                "title": event["title"],
                "activities": [
                    {"name": activity["name"], "prompt_preview": activity["prompt"][:100] + "..."}
                    for activity in event["activities"]
                ]
            }
            for event in HISTORICAL_EVENTS
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "active_video_tasks": len(task_status_store),
        "active_image_tasks": len(flux_task_store)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)