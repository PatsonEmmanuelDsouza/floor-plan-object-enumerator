# Floor Plan Object Enumerator 🛋️

This project uses a YOLOv8 model to detect and count furniture and other objects in floor plan images. The service is exposed via a FastAPI backend, which allows users to upload a floor plan and receive an annotated image along with a distinct count of the objects found.

The system is designed for continuous improvement, saving all predictions in a format ready for import into Label Studio to easily retrain and enhance the model's accuracy.



---

## Features

* **Object Detection:** Identifies and labels various furniture classes in floor plan images.
* **API Endpoint:** A simple FastAPI endpoint to handle image uploads and return predictions.
* **Object Counting:** Provides a count of distinct object classes found in the image.
* **Ready for Retraining:** Automatically saves uploaded images and prediction data (`predictions.json`) for easy re-labeling and model improvement in Label Studio.

---

## Model Details

* **Model Used:** YOLOv8m (`yolov8m.pt`)
* **Training:** The model was trained on a custom dataset using Google Colab.
* **Location:** The best-performing model is saved in the project at `models/best.pt`.

---

## Getting Started

To run the FastAPI server on your local environment, follow these steps.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install dependencies (when I publish it...):**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the server:**
    Use `uvicorn` to start the application. The `--reload` flag will automatically restart the server when you make code changes.
    ```bash
    uvicorn app.main:app --reload
    ```

4.  **View the API Docs:**
    Once the server is running, you can view the interactive API documentation by navigating to **http://localhost:8000/docs** in your browser.

---

## API Usage

### `POST /return-floorplan-objects`

This is the main endpoint for analyzing a floor plan.

* **Function:** Accepts a floor plan image file for object detection.
* **Input:** An image file (e.g., `.jpg`, `.png`).
* **Process:**
    1.  The YOLO model predicts objects in the uploaded image.
    2.  An annotated version of the image with bounding boxes and labels is created and saved.
    3.  A distinct count of each furniture class is calculated.
* **Output:** A JSON response containing:
    * The URI where the newly labeled image is stored.
    * A count of the distinct furniture classes found in the image.

---

## Continuous Improvement & Retraining

This project is built with a feedback loop in mind.

* **Local Image Storage:** Every image uploaded for prediction is saved locally. This creates an expanding dataset of real-world examples.
* **Label Studio Integration:** A `predictions.json` file is continuously updated with every prediction made. This file is formatted specifically for Label Studio. You can easily import it to review the model's predictions, correct any mistakes, and use this curated data to retrain and improve the YOLO model over time.