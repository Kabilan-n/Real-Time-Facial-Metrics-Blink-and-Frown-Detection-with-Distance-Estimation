# Face Analysis

## The objective here is to:

 Build a lightweight, on-device application that performs blink detection, distance  
 estimation, and frown detection in real-time, while measuring CPU usage and memory  
 consumption with high accuracy

## Expected outcomes

1. Count user’s blink: Implemented both Deep learning method as well as geometric method.  
2. Estimate user distance: added in the implementation  
3. Count frowns: added in the implementation  
4. on-screen overlay: the above things are added as an overlay to the output.  
5. Run pipeline for two mins: Added the output video along with the repo  
6.  Log CPU and memory usage once per second (average, peak) and calculate FPS: Logged along with the above recorded output and FPS included in the overlay  
7. Accuracy: Since there is no ground truth for the above demo. Manually verified almost 90% of the Blinks are captured. Distance from the screen is calculated based on the focal length approximation. Frown counter also works based on the provided threshold that needs to be adjusted or calibrated.   
8. Optimisation: The pipeline 1, that is blink detection using geometric method is less memory and CPU usage consumption but greater than 20%. The pipeline 2, since it uses a transformer model (advisable to use GPU) consumes high memory and CPU usage.   
9. Latency is \<100ms for pipeline 1, that is with geometric blink detection method.

## Setup and build instruction:

* Create a virtual environment  
* Install the required packages from requirements.txt  
* Run the python file: python ./webcam\_blink\_frown\_geo\_log.py or python ./webcam\_blink\_frown\_dl\_log.py  
* The output video and log files will be generated at the end. 

## Method used in each sub problem. 

### Blink detection:

For blink detection two methods have been implemented in this, 1\. Geometry based method. 2\. ViT based method.

#### In Geometry based method

The eye blinks are detected by calculating the Eye Aspect Ratio (EAR) based on facial landmarks. The following steps outline the process:

**Facial Landmark Extraction:** Utilize Dlib's shape predictor to extract facial landmarks.

**EAR Calculation:** The Eye Aspect Ratio is calculated using the formula:EAR= (∣∣p2−p6∣∣+∣∣p3−p5∣∣)/2⋅∣∣p1−p4∣∣, where p1 to p6 are points on the eye contour.

**Thresholding for Eye Closure:** If the EAR value falls below a predefined threshold (blink\_thresh), it indicates that the eye is closed.

**Average EAR Assessment:** Calculate the EAR for both eyes and compute the average EAR to enhance detection robustness.

**Frame Counting:** If the average EAR is less than the blink\_thresh, increment a frame counter. Once the frame counter reaches a specified number of consecutive frames (succ\_frame), increment the blink count. After a blink is detected, reset the frame counter.

#### 2\. ViT based method

**Eye Landmark Identification:**

The Dlib facial landmark predictor identifies key points corresponding to the left and right eyes. The indices for these landmarks are organized as follows:

* Left Eye: Landmarks are indexed from L\_start to L\_end.  
* Right Eye: Landmarks are indexed from R\_start to R\_end.  
  **Extraction of Eye Regions:**  
  Using the identified eye landmarks, bounding boxes are generated for each eye with the function mark\_eye\_landmark. These bounding boxes are slightly scaled to ensure the entire eye region is captured without clipping.  
  **Eye State Classification:**  
  The extracted eye regions are fed into a pre-trained transformer model from Hugging Face:  
  Model Used: **dima806/closed\_eyes\_image\_detection**  
  The AutoImageProcessor preprocesses the eye region to prepare it for the model, which predicts whether each eye is open (classified as openEye) or closed (classified as closeEye) based on the output logits.  
  **Eye State Transition Logic:**  
  Eye State Tracking  
  The states of the left and right eyes are monitored using a variable called previous\_eye\_state, which stores the previous frame's eye status (open or closed).  
  State Transition Rules  
  A transition from openEye to closeEye marks the beginning of a blink cycle. A transition from closeEye back to openEye indicates the completion of the blink cycle. The variable blink\_cycle\_detected ensures that both eyes successfully complete their respective blink cycles before counting it.  
  **Blink Counting Criteria:**  
  Conditions for Counting a Blink. A blink is recorded under the following conditions:  
* Both eyes transition from openEye to closeEye.  
* Both eyes then transition back to openEye from closeEye.  
  **Incrementing Blink Count:**  
  Once both eyes complete their state transitions, the blink\_count is incremented. Subsequently, the states within blink\_cycle\_detected are reset to False to prepare for the next potential blink cycle.

  ### Distance estimation:

  **Distance Estimation Formula**  
  The distance from the camera to a detected face is estimated using the pinhole camera model, which correlates the real-world dimensions of an object with its dimensions on the camera's image plane. The formula used for this estimation is:  
  Face Distance=(KNOWN\_FACE\_WIDTH×FOCAL\_LENGTH)/Face Width in Pixels  
  Parameters:  
  **KNOWN\_FACE\_WIDTH (160.0 mm)**: The average real-world width of a human face.  
  **FOCAL\_LENGTH (650 pixels)**: The calibrated focal length of the camera. (approx.)  
  **Face Width in Pixels:** The width of the detected face in the image.  
  A larger face size in the image indicates that the face is closer to the camera, resulting in a smaller calculated distance. Conversely, if the face appears smaller, the formula will yield a larger distance.  
  **Distance Threshold**  
  The proximity threshold is defined as follows:  
  DISTANCE\_THRESHOLD=508.0 mm  
  Faces that are closer than this threshold are considered to be in close proximity.  
  **Distance Event Counters**  
  The system tracks two key metrics related to the estimated face distance:  
  (a) distance\_less\_than\_threshold\_count  
  This counter records the number of instances where the detected face is closer than the defined distance threshold.  
  Logic:  
  For each frame:  
  If the calculated face\_distance is less than DISTANCE\_THRESHOLD, increment the counter.  
  (b) transition\_to\_close\_distance\_count  
  This counter monitors transitions where the detected face shifts from being farther than the threshold to being closer.  
  Logic:  
  A flag named previous\_distance\_above\_threshold is used to track whether the face was above the threshold in the previous frame. If the face was farther in the last frame but is closer in the current frame: Increment the transition\_to\_close\_distance\_count and update the flag to reflect the new state.

  ### Frown detection

  **Facial Landmark Detection:**  
  The code uses Dlib's facial landmark predictor to identify 68 key points on the face. These landmarks provide a detailed representation of facial features.  
  For frown detection, three specific landmarks are of interest point 21, 22 and 28  
  **Distance Calculation:**  
  The Euclidean distances between these landmarks are calculated to determine the facial expression:  
* Distance between points 22 and 28 (**dist\_22\_28**): This measures the vertical distance from the left corner of the eyebrow to the top nose.  
* Distance between points 21 and 22 (**dist\_21\_22**): This measures the horizontal distance between the two eyebrows.  
* Distance between points 21 and 28 (dist\_21\_28): This measures the distance from the right corner of the eyebrow to top nose.  
  **Thresholds for Frowning:**  
  The code defines specific threshold values for these distances that indicate a frown:  
* AVG\_FROWN\_DISTANCE\_22\_28: Threshold for the distance between points 22 and 28\.  
* AVG\_FROWN\_DISTANCE\_22\_21: Threshold for the distance between points 22 and 21\.  
* AVG\_FROWN\_DISTANCE\_21\_22: Threshold for the distance between points 21 and 22\.  
  These thresholds are set based on empirical observations of typical frowning distances.  
  **Frown Detection Logic**  
  The current frown state is determined by checking if any of the calculated distances fall below the defined thresholds:     
  If the current\_frown\_state is True and the previous\_frown\_state is False, it indicates that a frown has just been detected, and the frown count is incremented.

## System configuration:

* CPU: intel core Ultra 9  
* Core count: 6  
* OS: Windows 11  
* RAM: 15 GB

## Performance logs: 

	Attached in the repo

## Outline any techniques you used to reduce the above resource usage and optimise the solution further. What trade-offs need to be considered during optimization?

I felt using the Deep Learning method for a kind of task that can be solved geometrically is an overkill. Thus using geometric methods reduced the resource usage. And to make it still more optimised, this came thing can be implemented in C++, that reduces extra layer of conversion from assembly language. 

## Given more time, what would you do to solve the additional challenges mentioned in this test?

Given more time I would look into literature and identify computationally efficient techniques for the given tasks, and also work on containerising it with docker. Thus easily deployed across different platforms (Mac, Windows or Linux)

 