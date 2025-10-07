"""
    FastAPI application that defines endpoints that will enable a user to pass a floor plan image and get a response of the number of distinct funiture itmes in that image along with a labelled image with bounding boxes.
"""
from fastapi import FastAPI, HTTPException, Request, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
from fastapi import UploadFile
import os
import imghdr
from PIL import Image
import io
from .baseModels import FloorplanResponse
from collections import Counter
from datetime import datetime
from contextlib import asynccontextmanager
import uuid
import logging


# ----- startup function -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logging.info(" YOLO - Loading YOLO model on startup")
    get_yolo_model()
    # The server is now ready to accept requests
    yield
    # Code to run on shutdown (optional)
    logging.info("Server shutting down")

app = FastAPI(
    title="FloorPlan object detection model",
    description="An API that allows you to upload an image and serves the user with an image of detected objects along with the predicted count of all detected objects in that image",
    lifespan=lifespan
)


# ----- constants -----
MODEL_PATH = "models/best.pt"
model = None
PREDICTED_IMAGE_DIRECTORY = os.getenv('PREDICTED_IMAGE_DIRECTORY')
os.makedirs(PREDICTED_IMAGE_DIRECTORY, exist_ok=True)
LOG_DIR = os.getenv("LOGS_DIR","logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- 2. Set up Logging ---
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "app.log")

# Configure the root logger
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler(log_file_path), # Log to a file
        logging.StreamHandler()            # Log to the console
    ]
)

# ----- helper functions -----
def get_yolo_model():
    """
    Loads the YOLO model so that we can execute predictions
    """
    global model
    if model is None:
        logging.info("MODEL INFO - Model was not loaded yet. Loading now...")
        if not os.path.exists(MODEL_PATH):
            logging.error(f"Model folder not found at {MODEL_PATH}")
            return None
        try:    
            model = YOLO(MODEL_PATH)
            logging.info("MODEL INFO - Model loaded successfully.")
        except Exception as e:
            logging.exception(f"Could not load model. The following exception occured:")
    return model

def getCountOfItems(results: list)->dict:
    """
    a method that will return a dict that specifies the distinct furniture pieces found in a floorplan image
    """
    # Get the dictionary of class names
    class_names = results[0].names
    # Get the list of predicted class IDs
    predicted_class_ids = results[0].boxes.cls.tolist()
    # Map the class IDs to class names
    predicted_class_names = [class_names[int(id)] for id in predicted_class_ids]
    # Count the occurrences of each class name
    object_counts = dict(Counter(predicted_class_names))
    return object_counts  

def save_predicted_yolo_image(results: list):
    """
    Method that enables saving floor plan image
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{uuid.uuid4()}-{current_time}.jpg"
    save_path = f"{PREDICTED_IMAGE_DIRECTORY}/{filename}"
    results[0].save(filename=str(save_path))
    return filename

# ----- API Methods -----

@app.get('/')
def home():
    return {'message': "welcome to this tool!"}


@app.get("/predicted-images/{filename}", name = 'get_image', tags=["Images Folder"])
async def get_image(filename: str):
    """
    Custom get function to get the image stored on backend when you look up the file
    """
    logging.info(f"IMAGE READ - attempting to serve the image '{filename}'")
    file_path = os.path.join(PREDICTED_IMAGE_DIRECTORY, filename)
    
    if not os.path.exists(file_path):
        logging.error(f"The image {filename} was not found on disk!")
        raise HTTPException(status_code=404, detail="Image not found.")
    
    return FileResponse(file_path)


@app.post("/return-floorplan-objects", response_model=FloorplanResponse)
async def analyze_floorplan_image(request: Request, file: UploadFile = File(...)):
    """
    This endpoint accepts an image file, verifies it, saves it,
    and returns the detected objects within the floorplan.
    """
    logging.info("Attempting to get information on the objects of a floorplan.")
    result = FloorplanResponse(
        status="error",
        image_url="-",
        result={}
    )
    
    # 1. --- Verify the Uploaded File is an Image ---
    # Check the content type provided by the browser
    if not file.content_type.startswith("image/"):
        logging.error(f"The image uploaded was not an image file!")
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    # Read the file content to perform a more robust check
    contents = await file.read()
    
    # Use imghdr to determine the image type from its content
    image_type = imghdr.what(None, h=contents)
    if image_type not in ["jpeg", "png", "gif", "bmp"]: # Add other types if needed
        logging.error(f"The image uploaded was not an image file!")
        raise HTTPException(status_code=400, detail="Invalid image file content.")

    # TODO: idea; define a quick image model to check if the upload is a floorplan or not
    
    if not model:
        logging.error(f"YOLO model is not laoded!")
        return result
    try:
        # Convert the bytes into a PIL Image
        pil_image = Image.open(io.BytesIO(contents))
        # Run inference on the PIL image
        prediction_results = model(pil_image)
        results = getCountOfItems(prediction_results)
        result.result = results
        result.status="success"
        filename = save_predicted_yolo_image(prediction_results)        
        resource_url = request.url_for('get_image', filename=filename)    
        result.image_url = str(resource_url)
        return result
    
    except Exception as e:
        # Handle potential errors during model inference
        logging.exception(f"Was not able to analyze the image due to the following exception: ")
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
