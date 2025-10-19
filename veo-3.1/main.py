import os
import uuid
import requests
import json
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from pathlib import Path
from supabase import create_client, Client
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Historical Video Generator", version="1.0.0")

# Create necessary directories
Path("outputs").mkdir(exist_ok=True)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Veo API configuration
API_KEY = os.getenv("API_KEY")
API_BASE_URL = "https://api.kie.ai"

# Store task status in memory (in production, use Redis or database)
task_status_store: Dict[str, Dict[str, Any]] = {}

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
    {
        "id": "salt_march",
        "title": "Salt March - The Freedom Movement",
        "activities": [
            {
                "name": "Growing March Numbers",
                "prompt": "A vlogger walks along the march route with growing crowds following. The vlogger turns to show the expanding group of freedom fighters. More people join the march in procession. Collective energy and determination visible. The vlogger gestures showing unity of the movement. Camera captures vlogger surrounded by growing crowd of marchers. Vertical mobile format."
            }
        ]
    },
    {
        "id": "ashoka",
        "title": "Emperor Ashoka - The Beloved of the Gods",
        "activities": [
            {
                "name": "The Edicts of Ashoka",
                "prompt": "A vlogger stands before the famous Ashoka pillars with carved edicts visible. Stone inscriptions and carved text clearly displayed. The vlogger traces their hand over the ancient edicts reading them aloud. Close-up of intricate stone carving work. The vlogger explains the moral and administrative messages. Late afternoon light creates shadows emphasizing the carving. Camera captures vlogger's reverent interaction with the stone. Vertical mobile format."
            }
        ]
    },
    {
        "id": "taj_mahal",
        "title": "Taj Mahal - Monument to Love",
        "activities": [
            {
                "name": "Death of Mumtaz Mahal",
                "prompt": "A vlogger stands at a historical palace chamber or memorial space honoring Mumtaz Mahal. Flowers and tributes visible. The vlogger speaks solemnly about her death and Shah Jahan's grief. Emotional, respectful atmosphere. Murals or paintings depicting the moment visible. Soft contemplative lighting. Camera captures vlogger's respectful expression. Vertical mobile format."
            },
        ]
    },
    {
        "id": "indus_valley",
        "title": "Indus Valley Civilization - Ancient Mysteries",
        "activities": [
            {
                "name": "Discovery of the Civilization",
                "prompt": "A vlogger stands in an archaeological museum displaying initial discoveries from Indus Valley. Ancient seals and pottery fragments in glass cases. The vlogger explains the archaeological discovery process. Museum lighting illuminates artifacts. The vlogger points to breakthrough discoveries. Historical excavation photographs on walls. Camera captures vlogger among the artifacts. Vertical mobile format."
            }
        ]
    },
    {
        "id": "mughal_empire",
        "title": "Mughal Empire - Splendor and Majesty",
        "activities": [
            {
                "name": "Court and Administration",
                "prompt": "A vlogger stands in a historical Mughal court setting or museum reconstruction. Throne platforms and royal court architecture visible. The vlogger stands in the center explaining the court proceedings. Intricate architectural details and carvings surround. The vlogger demonstrates the governance structure and court hierarchy. Dramatic lighting emphasizes the grandeur. Camera captures vlogger in the royal court setting. Vertical mobile format."
            }
        ]
    },
    {
        "id": "maurya_empire",
        "title": "Maurya Empire - The Great Dynasty",
        "activities": [
            {
                "name": "Expansion and Military Might",
                "prompt": "A vlogger stands at a historical map display or at a site showing Maurya empire territorial extent. Vast landscape representing empire boundaries. The vlogger gestures showing the empire's expansion across the continent. Military monuments and fortifications visible. The vlogger explains military strategies and conquests. Golden hour creates dramatic lighting. Camera captures vlogger gesturing to show territorial expansion. Vertical mobile format."
            }
        ]
    },
    {
        "id": "chola_dynasty",
        "title": "Chola Dynasty - Maritime Traders",
        "activities": [
            {
                "name": "Bronze Sculpture Mastery",
                "prompt": "A vlogger stands in a museum displaying famous Chola bronze sculptures. Dancing Shiva and other bronze masterpieces displayed. The vlogger marvels at the artistic perfection. Museum lighting illuminates the sculptures. The vlogger points to artistic details and techniques. The vlogger demonstrates dance poses shown in sculptures. Camera captures vlogger interacting with bronze art. Vertical mobile format."
            }
        ]
    },
    {
        "id": "independence_day",
        "title": "Independence Day - Freedom Achieved",
        "activities": [
            {
                "name": "Rise of Freedom Fighters",
                "prompt": "A vlogger stands at memorials of famous freedom fighters with their statues and plaques. Revolutionary warriors' monuments visible. The vlogger points to different freedom fighters' contributions. Heroic sculptures and inscriptions displayed. The vlogger speaks passionately about their sacrifices. Golden light creates heroic atmosphere. Camera captures vlogger before freedom fighter memorials. Vertical mobile format."
            }
        ]
    }
]

class VideoGenerationResponse(BaseModel):
    task_id: str
    status: str
    video_url: Optional[str] = None
    download_url: Optional[str] = None
    message: str

def get_event_by_id(event_id: str):
    """Find event by ID"""
    for event in HISTORICAL_EVENTS:
        if event["id"] == event_id:
            return event
    return None

async def upload_image_to_supabase(image_file: UploadFile) -> str:
    """Upload image to Supabase Storage and return public URL"""
    try:
        # Generate unique filename
        file_extension = image_file.filename.split('.')[-1] if image_file.filename else 'jpg'
        filename = f"{uuid.uuid4()}.{file_extension}"
        
        # Read file content
        content = await image_file.read()
        
        # Upload to Supabase Storage
        bucket_name = "images"
        
        response = supabase.storage.from_(bucket_name).upload(
            file=content,
            path=filename,
            file_options={"content-type": image_file.content_type}
        )
        
        # Get public URL
        public_url = supabase.storage.from_(bucket_name).get_public_url(filename)
        
        logger.info(f"Image uploaded to Supabase: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"Supabase upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

def call_veo_api(prompt: str, image_url: str, event_name: str) -> dict:
    """Call Veo 3.1 API to generate video"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "imageUrls": [image_url],
        "model": "veo3_fast",
        "generationType": "REFERENCE_2_VIDEO",
        "aspectRatio": "16:9",
        "enableTranslation": True,
        "watermark": f"Historical {event_name}"
    }
    
    logger.info(f"Calling Veo API with image URL: {image_url}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/veo/generate",
            headers=headers,
            json=payload,
            timeout=30
        )
        logger.info(f"API Response status: {response.status_code}")
        logger.info(f"API Response content: {response.text}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video generation API error: {e}")

def check_veo_task_status(task_id: str) -> Dict[str, Any]:
    """Check Veo task status using the record-info endpoint"""
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/veo/record-info",
            headers=headers,
            params={"taskId": task_id},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logger.error(f"Status check failed: {response.status_code} - {response.text}")
            return {"code": response.status_code, "msg": "Failed to check status"}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Status check request failed: {e}")
        return {"code": 500, "msg": str(e)}

def download_video(video_url: str, output_path: str) -> bool:
    """Download generated video to outputs folder"""
    try:
        response = requests.get(video_url, timeout=60)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Video downloaded to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Video download failed: {e}")
        return False

async def monitor_video_generation(task_id: str):
    """Background task to monitor video generation and update status"""
    max_attempts = 60  # Check for 10 minutes (60 * 10 seconds)
    attempt = 0
    
    # Initialize task status
    task_status_store[task_id] = {
        "status": "processing",
        "video_url": None,
        "download_url": None,
        "message": "Video generation in progress"
    }
    
    while attempt < max_attempts:
        try:
            logger.info(f"Checking video status for task {task_id}, attempt {attempt + 1}")
            status_data = check_veo_task_status(task_id)
            
            if status_data.get("code") == 200:
                data = status_data.get("data", {})
                success_flag = data.get("successFlag", 0)
                
                if success_flag == 1:  # Success
                    response_data = data.get("response", {})
                    video_urls = response_data.get("resultUrls", [])
                    
                    if video_urls:
                        video_url = video_urls[0]
                        output_filename = f"{task_id}.mp4"
                        output_path = f"outputs/{output_filename}"
                        
                        # Download the video
                        if download_video(video_url, output_path):
                            task_status_store[task_id] = {
                                "status": "completed",
                                "video_url": video_url,
                                "download_url": f"/download-video/{task_id}",
                                "message": "Video generation completed successfully"
                            }
                            logger.info(f"Video generation completed for task {task_id}")
                            break
                        else:
                            task_status_store[task_id] = {
                                "status": "completed",
                                "video_url": video_url,
                                "download_url": None,
                                "message": "Video generated but download failed"
                            }
                            logger.warning(f"Video generated but download failed for task {task_id}")
                            break
                    else:
                        task_status_store[task_id] = {
                            "status": "error",
                            "video_url": None,
                            "download_url": None,
                            "message": "No video URLs found in response"
                        }
                        logger.error(f"No video URLs found for task {task_id}")
                        break
                        
                elif success_flag == 2:  # Failed
                    error_message = data.get("errorMessage", "Unknown error")
                    task_status_store[task_id] = {
                        "status": "failed",
                        "video_url": None,
                        "download_url": None,
                        "message": f"Video generation failed: {error_message}"
                    }
                    logger.error(f"Video generation failed for task {task_id}: {error_message}")
                    break
                else:  # Still processing (success_flag == 0)
                    task_status_store[task_id] = {
                        "status": "processing",
                        "video_url": None,
                        "download_url": None,
                        "message": f"Video generation in progress... ({attempt + 1}/{max_attempts})"
                    }
            else:
                error_msg = status_data.get("msg", "Unknown error")
                task_status_store[task_id] = {
                    "status": "error",
                    "video_url": None,
                    "download_url": None,
                    "message": f"Status check error: {error_msg}"
                }
                logger.error(f"Status check error for task {task_id}: {error_msg}")
                break
                
        except Exception as e:
            logger.error(f"Error monitoring task {task_id}: {e}")
            task_status_store[task_id] = {
                "status": "error",
                "video_url": None,
                "download_url": None,
                "message": f"Monitoring error: {str(e)}"
            }
        
        # Wait before next check
        await asyncio.sleep(10)
        attempt += 1
    
    # If max attempts reached and still processing
    if attempt >= max_attempts and task_status_store.get(task_id, {}).get("status") == "processing":
        task_status_store[task_id] = {
            "status": "timeout",
            "video_url": None,
            "download_url": None,
            "message": "Video generation timeout after 10 minutes"
        }
        logger.error(f"Video generation timeout for task {task_id}")

@app.get("/")
async def root():
    return {"message": "Historical Video Generator API", "version": "1.0.0"}

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

@app.post("/generate-video", response_model=VideoGenerationResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    event_id: str = Form(...),
    activity_index: int = Form(0),
    image: UploadFile = File(...)
):
    """
    Generate historical video with user's face and return video link
    """
    # Validate image file
    if not image.content_type.startswith('image/'):
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
        # Upload image to Supabase
        logger.info("Uploading image to Supabase...")
        image_url = await upload_image_to_supabase(image)
        
        # Call Veo API
        logger.info("Calling Veo API...")
        api_response = call_veo_api(
            prompt=activity["prompt"],
            image_url=image_url,
            event_name=event["title"]
        )
        
        if api_response.get("code") != 200:
            raise HTTPException(
                status_code=500, 
                detail=f"API Error: {api_response.get('msg', 'Unknown error')}"
            )
        
        task_id = api_response["data"]["taskId"]
        
        # Start background task to monitor video generation
        background_tasks.add_task(monitor_video_generation, task_id)
        
        return VideoGenerationResponse(
            task_id=task_id,
            status="processing",
            message="Video generation started successfully. Use the status endpoint to check progress."
        )
        
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-status/{task_id}")
async def get_video_status(task_id: str):
    """Check video generation status and get video link"""
    task_status = task_status_store.get(task_id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    return VideoGenerationResponse(
        task_id=task_id,
        status=task_status["status"],
        video_url=task_status.get("video_url"),
        download_url=task_status.get("download_url"),
        message=task_status["message"]
    )

@app.get("/download-video/{task_id}")
async def download_generated_video(task_id: str):
    """Download generated video file"""
    video_path = f"outputs/{task_id}.mp4"
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found or still processing")
    
    return FileResponse(
        video_path,
        media_type='video/mp4',
        filename=f"historical_video_{task_id}.mp4"
    )

@app.get("/videos")
async def list_generated_videos():
    """List all generated videos"""
    videos = []
    for filename in os.listdir("outputs"):
        if filename.endswith(".mp4"):
            file_path = f"outputs/{filename}"
            stat = os.stat(file_path)
            task_id = filename.replace(".mp4", "")
            videos.append({
                "filename": filename,
                "task_id": task_id,
                "size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "download_url": f"/download-video/{task_id}",
                "status": task_status_store.get(task_id, {}).get("status", "unknown")
            })
    
    return {"videos": videos}

@app.get("/task/{task_id}")
async def get_task_details(task_id: str):
    """Get detailed Veo task information"""
    status_data = check_veo_task_status(task_id)
    return status_data

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(task_status_store)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)