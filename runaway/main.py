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

app = FastAPI(title="Historical Media Generator", version="3.0.0")

# Create necessary directories
Path("outputs").mkdir(exist_ok=True)
Path("images").mkdir(exist_ok=True)
Path("runway_videos").mkdir(exist_ok=True)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# API Keys
VEO_API_KEY = os.getenv("VEO_API_KEY")
FLUX_API_KEY = os.getenv("FLUX_API_KEY")
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
API_BASE_URL = "https://api.kie.ai"

# Store task status in memory
task_status_store: Dict[str, Dict[str, Any]] = {}
flux_task_store: Dict[str, Dict[str, Any]] = {}
runway_task_store: Dict[str, Dict[str, Any]] = {}

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
    # ... (other events remain the same)
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

class RunwayGenerationResponse(BaseModel):
    task_id: str
    status: str
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    download_url: Optional[str] = None
    message: str

class RunwayGenerationRequest(BaseModel):
    prompt: str
    duration: int = 5
    quality: str = "720p"
    aspect_ratio: str = "16:9"
    watermark: str = ""

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

# ==================== RUNWAY API FUNCTIONS ====================

def call_runway_api(prompt: str, image_url: Optional[str] = None, **kwargs) -> dict:
    """Call Runway API to generate videos"""
    headers = {
        "Authorization": f"Bearer {RUNWAY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "duration": kwargs.get("duration", 5),
        "quality": kwargs.get("quality", "720p"),
        "aspectRatio": kwargs.get("aspect_ratio", "16:9"),
        "waterMark": kwargs.get("watermark", "")
    }
    
    # Add image URL for image-to-video generation
    if image_url:
        payload["imageUrl"] = image_url
    
    logger.info(f"Calling Runway API with payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/runway/generate",
            headers=headers,
            json=payload,
            timeout=30
        )
        logger.info(f"Runway API Response status: {response.status_code}")
        logger.info(f"Runway API Response content: {response.text}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Runway API request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Runway video generation API error: {e}")

def call_runway_extend_api(task_id: str, prompt: str, **kwargs) -> dict:
    """Call Runway API to extend existing videos"""
    headers = {
        "Authorization": f"Bearer {RUNWAY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "taskId": task_id,
        "prompt": prompt,
        "quality": kwargs.get("quality", "720p")
    }
    
    logger.info(f"Calling Runway Extend API with payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/runway/extend",
            headers=headers,
            json=payload,
            timeout=30
        )
        logger.info(f"Runway Extend API Response status: {response.status_code}")
        logger.info(f"Runway Extend API Response content: {response.text}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Runway Extend API request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Runway video extension API error: {e}")

def check_runway_task_status(task_id: str) -> Dict[str, Any]:
    """Check Runway task status using the record-detail endpoint"""
    headers = {
        "Authorization": f"Bearer {RUNWAY_API_KEY}"
    }
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/runway/record-detail",
            headers=headers,
            params={"taskId": task_id},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logger.error(f"Runway status check failed: {response.status_code} - {response.text}")
            return {"code": response.status_code, "msg": "Failed to check status"}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Runway status check request failed: {e}")
        return {"code": 500, "msg": str(e)}

def download_runway_video(video_url: str, output_path: str) -> bool:
    """Download generated Runway video to runway_videos folder"""
    try:
        response = requests.get(video_url, timeout=60)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Runway video downloaded to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Runway video download failed: {e}")
        return False

async def monitor_runway_generation(task_id: str):
    """Background task to monitor Runway video generation and update status"""
    max_attempts = 60  # Check for 10 minutes (60 * 10 seconds)
    attempt = 0
    
    # Initialize task status
    runway_task_store[task_id] = {
        "status": "processing",
        "video_url": None,
        "thumbnail_url": None,
        "download_url": None,
        "message": "Video generation in progress"
    }
    
    while attempt < max_attempts:
        try:
            logger.info(f"Checking Runway video status for task {task_id}, attempt {attempt + 1}")
            status_data = check_runway_task_status(task_id)
            
            if status_data.get("code") == 200:
                data = status_data.get("data", {})
                state = data.get("state", "wait")
                
                if state == "success":  # Success
                    video_info = data.get("videoInfo", {})
                    video_url = video_info.get("videoUrl")
                    thumbnail_url = video_info.get("imageUrl")
                    
                    if video_url:
                        output_filename = f"{task_id}.mp4"
                        output_path = f"runway_videos/{output_filename}"
                        
                        # Download the video
                        if download_runway_video(video_url, output_path):
                            runway_task_store[task_id] = {
                                "status": "completed",
                                "video_url": video_url,
                                "thumbnail_url": thumbnail_url,
                                "download_url": f"/download-runway-video/{task_id}",
                                "message": "Video generation completed successfully"
                            }
                            logger.info(f"Runway video generation completed for task {task_id}")
                            break
                        else:
                            runway_task_store[task_id] = {
                                "status": "completed",
                                "video_url": video_url,
                                "thumbnail_url": thumbnail_url,
                                "download_url": None,
                                "message": "Video generated but download failed"
                            }
                            logger.warning(f"Runway video generated but download failed for task {task_id}")
                            break
                    else:
                        runway_task_store[task_id] = {
                            "status": "error",
                            "video_url": None,
                            "thumbnail_url": None,
                            "download_url": None,
                            "message": "No video URL found in response"
                        }
                        logger.error(f"No video URL found for task {task_id}")
                        break
                        
                elif state == "fail":  # Failed
                    error_message = data.get("failMsg", "Unknown error")
                    runway_task_store[task_id] = {
                        "status": "failed",
                        "video_url": None,
                        "thumbnail_url": None,
                        "download_url": None,
                        "message": f"Video generation failed: {error_message}"
                    }
                    logger.error(f"Runway video generation failed for task {task_id}: {error_message}")
                    break
                    
                else:  # Still processing (wait, queueing, generating)
                    status_messages = {
                        "wait": "Task is waiting to start",
                        "queueing": "Task is in queue",
                        "generating": "Video is being generated"
                    }
                    
                    runway_task_store[task_id] = {
                        "status": state,
                        "video_url": None,
                        "thumbnail_url": None,
                        "download_url": None,
                        "message": f"{status_messages.get(state, 'Processing')}... ({attempt + 1}/{max_attempts})"
                    }
            else:
                error_msg = status_data.get("msg", "Unknown error")
                runway_task_store[task_id] = {
                    "status": "error",
                    "video_url": None,
                    "thumbnail_url": None,
                    "download_url": None,
                    "message": f"Status check error: {error_msg}"
                }
                logger.error(f"Runway status check error for task {task_id}: {error_msg}")
                break
                
        except Exception as e:
            logger.error(f"Error monitoring Runway task {task_id}: {e}")
            runway_task_store[task_id] = {
                "status": "error",
                "video_url": None,
                "thumbnail_url": None,
                "download_url": None,
                "message": f"Monitoring error: {str(e)}"
            }
        
        # Wait before next check
        await asyncio.sleep(10)
        attempt += 1
    
    # If max attempts reached and still processing
    if attempt >= max_attempts and runway_task_store.get(task_id, {}).get("status") in ["wait", "queueing", "generating"]:
        runway_task_store[task_id] = {
            "status": "timeout",
            "video_url": None,
            "thumbnail_url": None,
            "download_url": None,
            "message": "Video generation timeout after 10 minutes"
        }
        logger.error(f"Runway video generation timeout for task {task_id}")

# ==================== RUNWAY API ENDPOINTS ====================

@app.post("/runway/generate-video", response_model=RunwayGenerationResponse)
async def generate_runway_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    duration: int = Form(5),
    quality: str = Form("720p"),
    aspect_ratio: str = Form("16:9"),
    watermark: str = Form(""),
    input_image: Optional[UploadFile] = File(None)
):
    """
    Generate video using Runway API (text-to-video or image-to-video)
    """
    input_image_url = None
    
    try:
        # Upload input image if provided (for image-to-video)
        if input_image:
            if not input_image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Input file must be an image")
            logger.info("Uploading input image to Supabase...")
            input_image_url = await upload_image_to_supabase(input_image)
        
        # Validate parameters
        if duration not in [5, 10]:
            raise HTTPException(status_code=400, detail="Duration must be 5 or 10 seconds")
        
        if quality not in ["720p", "1080p"]:
            raise HTTPException(status_code=400, detail="Quality must be 720p or 1080p")
        
        if quality == "1080p" and duration != 5:
            raise HTTPException(status_code=400, detail="1080p quality is only available for 5-second videos")
        
        # Call Runway API
        logger.info("Calling Runway API...")
        api_response = call_runway_api(
            prompt=prompt,
            image_url=input_image_url,
            duration=duration,
            quality=quality,
            aspect_ratio=aspect_ratio,
            watermark=watermark
        )
        
        if api_response.get("code") != 200:
            raise HTTPException(
                status_code=500, 
                detail=f"Runway API Error: {api_response.get('msg', 'Unknown error')}"
            )
        
        task_id = api_response["data"]["taskId"]
        
        # Start background task to monitor video generation
        background_tasks.add_task(monitor_runway_generation, task_id)
        
        return RunwayGenerationResponse(
            task_id=task_id,
            status="processing",
            message="Runway video generation started successfully. Use the status endpoint to check progress."
        )
        
    except Exception as e:
        logger.error(f"Runway video generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/runway/generate-historical-video", response_model=RunwayGenerationResponse)
async def generate_historical_runway_video(
    background_tasks: BackgroundTasks,
    event_id: str = Form(...),
    activity_index: int = Form(0),
    user_image: UploadFile = File(...),
    duration: int = Form(5),
    quality: str = Form("720p"),
    aspect_ratio: str = Form("16:9")
):
    """
    Generate historical video with user's face using Runway
    """
    # Validate image file
    if not user_image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate parameters
    if duration not in [5, 10]:
        raise HTTPException(status_code=400, detail="Duration must be 5 or 10 seconds")
    
    if quality == "1080p" and duration != 5:
        raise HTTPException(status_code=400, detail="1080p quality is only available for 5-second videos")
    
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
        runway_prompt = f"{activity['prompt']} The vlogger has the face and appearance of the person in the reference image."
        
        # Call Runway API for image-to-video generation
        logger.info("Calling Runway API for historical video...")
        api_response = call_runway_api(
            prompt=runway_prompt,
            image_url=user_image_url,
            duration=duration,
            quality=quality,
            aspect_ratio=aspect_ratio,
            watermark=f"Historical {event['title']}"
        )
        
        if api_response.get("code") != 200:
            raise HTTPException(
                status_code=500, 
                detail=f"Runway API Error: {api_response.get('msg', 'Unknown error')}"
            )
        
        task_id = api_response["data"]["taskId"]
        
        # Start background task to monitor video generation
        background_tasks.add_task(monitor_runway_generation, task_id)
        
        return RunwayGenerationResponse(
            task_id=task_id,
            status="processing",
            message=f"Historical video generation started for {event['title']}. Use the status endpoint to check progress."
        )
        
    except Exception as e:
        logger.error(f"Historical Runway video generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/runway/extend-video", response_model=RunwayGenerationResponse)
async def extend_runway_video(
    background_tasks: BackgroundTasks,
    original_task_id: str = Form(...),
    prompt: str = Form(...),
    quality: str = Form("720p")
):
    """
    Extend an existing Runway video
    """
    try:
        # Validate quality
        if quality not in ["720p", "1080p"]:
            raise HTTPException(status_code=400, detail="Quality must be 720p or 1080p")
        
        # Call Runway Extend API
        logger.info("Calling Runway Extend API...")
        api_response = call_runway_extend_api(
            task_id=original_task_id,
            prompt=prompt,
            quality=quality
        )
        
        if api_response.get("code") != 200:
            raise HTTPException(
                status_code=500, 
                detail=f"Runway Extend API Error: {api_response.get('msg', 'Unknown error')}"
            )
        
        task_id = api_response["data"]["taskId"]
        
        # Start background task to monitor video generation
        background_tasks.add_task(monitor_runway_generation, task_id)
        
        return RunwayGenerationResponse(
            task_id=task_id,
            status="processing",
            message="Video extension started successfully. Use the status endpoint to check progress."
        )
        
    except Exception as e:
        logger.error(f"Runway video extension failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runway/video-status/{task_id}")
async def get_runway_video_status(task_id: str):
    """Check Runway video generation status and get video link"""
    task_status = runway_task_store.get(task_id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    return RunwayGenerationResponse(
        task_id=task_id,
        status=task_status["status"],
        video_url=task_status.get("video_url"),
        thumbnail_url=task_status.get("thumbnail_url"),
        download_url=task_status.get("download_url"),
        message=task_status["message"]
    )

@app.get("/download-runway-video/{task_id}")
async def download_runway_video(task_id: str):
    """Download generated Runway video file"""
    video_path = f"runway_videos/{task_id}.mp4"
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found or still processing")
    
    return FileResponse(
        video_path,
        media_type='video/mp4',
        filename=f"runway_video_{task_id}.mp4"
    )

@app.get("/runway/videos")
async def list_runway_videos():
    """List all generated Runway videos"""
    videos = []
    for filename in os.listdir("runway_videos"):
        if filename.endswith(".mp4"):
            file_path = f"runway_videos/{filename}"
            stat = os.stat(file_path)
            task_id = filename.replace(".mp4", "")
            videos.append({
                "filename": filename,
                "task_id": task_id,
                "size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "download_url": f"/download-runway-video/{task_id}",
                "status": runway_task_store.get(task_id, {}).get("status", "unknown")
            })
    
    return {"videos": videos}

@app.get("/runway/task/{task_id}")
async def get_runway_task_details(task_id: str):
    """Get detailed Runway task information"""
    status_data = check_runway_task_status(task_id)
    return status_data

# ==================== EXISTING VEO AND FLUX ENDPOINTS ====================

# ... (Keep all your existing Veo and Flux API endpoints as they are)

@app.get("/")
async def root():
    return {
        "message": "Historical Media Generator API", 
        "version": "3.0.0",
        "features": ["Veo Video Generation", "Flux Image Generation", "Runway Video Generation"]
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
        "active_veo_tasks": len(task_status_store),
        "active_flux_tasks": len(flux_task_store),
        "active_runway_tasks": len(runway_task_store)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)