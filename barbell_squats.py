import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict
from models import db, UserExercise
import time
from flask import redirect, url_for
import os
import subprocess

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def gen_frames(user_id, rep_goal):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Initialize thresholds for beginner level
    thresholds = {
        'HIP_KNEE_VERT': {
            'NORMAL': (0, 30),
            'TRANS': (35, 65),
            'PASS': (70, 95)
        },
        'HIP_THRESH': [10, 60],
        'ANKLE_THRESH': 45,
        'KNEE_THRESH': [50, 70, 95],
        'OFFSET_THRESH': 50.0,
        'INACTIVE_THRESH': 15.0,
        'CNT_FRAME_THRESH': 50
    }

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        return

    state_tracker = {
        'state_seq': [],
        'start_inactive_time': time.perf_counter(),
        'start_inactive_time_front': time.perf_counter(),
        'INACTIVE_TIME': 0.0,
        'INACTIVE_TIME_FRONT': 0.0,
        'DISPLAY_TEXT': np.full((4,), False),
        'COUNT_FRAMES': np.zeros((4,), dtype=np.int64),
        'LOWER_HIPS': False,
        'INCORRECT_POSTURE': False,
        'prev_state': None,
        'curr_state': None,
        'SQUAT_COUNT': 0,
        'IMPROPER_SQUAT': 0,
        'squat_attempt': 0,
        'current_mistakes': {},
        'mistakes': []
    }

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # Calculate angles
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Display angle
                cv2.putText(frame, str(int(knee_angle)), 
                           tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Analyze form
                if knee_angle < 80:
                        feedback = "Squat deeper!"
                elif knee_angle > 140:
                        feedback = "Too high!"
                else:
                        feedback = "Good form!"

                cv2.putText(frame, feedback, (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Count reps
                if knee_angle < 80 and state_tracker['prev_state'] != 'down':
                    state_tracker['squat_attempt'] += 1
                    state_tracker['prev_state'] = 'down'
                elif knee_angle > 140 and state_tracker['prev_state'] == 'down':
                    state_tracker['SQUAT_COUNT'] += 1
                    state_tracker['prev_state'] = 'up'

                # Display rep count
                cv2.putText(frame, f'Reps: {state_tracker["SQUAT_COUNT"]}', (30, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Check if rep goal is reached
                if state_tracker['SQUAT_COUNT'] >= rep_goal:
                    # Save exercise data to database
                                new_exercise = UserExercise(
                                    user_id=user_id,
                        exercise_id=3,  # Barbell Squats ID
                                    total_reps=rep_goal,
                        rom_score=rep_goal,  # Assuming all reps were good for now
                        tut_score=0.0,  # Time under tension score
                                    count=rep_goal
                                )
                                db.session.add(new_exercise)
                                db.session.commit()
                break

            except Exception as e:
                print(f"Error during pose processing: {e}")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

def analyze_squat_video(video_path):
    """
    Analyzes squat video and provides feedback on form
    """
    print(f"Starting analysis of video: {video_path}")
    mp_pose = mp.solutions.pose
    
    # Path for processed video - store in the uploads/exercises directory
    timestamp = int(time.time())
    video_filename = f"squat_analysis_{timestamp}.mp4"
    
    # Get the app's UPLOAD_FOLDER from environment or default to uploads/exercises
    try:
        from app import app
        output_dir = app.config.get('UPLOAD_FOLDER', 'uploads/exercises')
    except Exception as e:
        print(f"Error getting app config: {e}")
        output_dir = 'uploads/exercises'
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, video_filename)
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return {"error": "Could not open video file"}
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video properties: {width}x{height}, {fps}fps, {total_frames} frames, duration: {duration:.2f}s")
    
    # Ensure we have valid video properties
    if width <= 0 or height <= 0 or fps <= 0:
        print("Invalid video properties detected")
        return {"error": "Invalid video properties"}
    
    # Set up video writer with same parameters as input
    target_width = min(width, 640)  # Cap width at 640px for performance
    target_height = int(height * (target_width / width))
    
    # Try different codecs if needed
    try:
        # First try H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
        # Check if writer opened successfully
        if not out.isOpened():
            print("Failed to open VideoWriter with avc1 codec, trying mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
            
            if not out.isOpened():
                print("Failed to open VideoWriter with mp4v codec, trying XVID")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    except Exception as e:
        print(f"Error setting up VideoWriter: {e}")
        return {"error": f"Failed to create output video: {e}"}
        
    if not out.isOpened():
        print("All codec attempts failed")
        return {"error": "Failed to create output video with any codec"}
    
    # For dynamic feedback tracking
    all_feedback = []  # Store all unique feedback messages for summary
    active_feedback = []  # Current active feedback messages displayed on screen
    correct_rep_count = 0  # Count of correct reps
    incorrect_rep_count = 0 # Count of incorrect reps
    feedback_duration_frames = int(fps * 2)  # Feedback stays for 2 seconds
    feedback_frame_counters = {}  # Track how long each feedback has been shown
    
    # Rep state tracking
    last_valid_landmarks = None
    in_squat = False  # Track if currently in squat position
    current_rep_feedback = [] # Feedback generated during the current rep
    
    # Rep detection thresholds
    MIN_KNEE_ANGLE = 80  # Below this is considered a squat
    MAX_KNEE_ANGLE = 160  # Above this is considered standing
    
    # Thresholds for specific feedback
    HIP_THRESH_BACKWARDS = 60  # Above this angle indicates bending backwards
    KNEE_ANKLE_DIST_THRESH = 0.2  # Threshold for knee falling over toe (increased from 0.1 to be much more lenient)
    MAX_SQUAT_DEPTH = 50  # Below this angle is too deep
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose:
        # Process every frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames_written = 0
        frame_count = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            # Resize frame for consistency
            frame_resized = cv2.resize(frame, (target_width, target_height))
            
            # Make a copy of the original frame for drawing
            display_frame = frame_resized.copy()
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            current_landmarks = None
            current_knee_angle = None
            rep_just_completed = False
            
            # Check if we have valid landmarks
            if results.pose_landmarks:
                last_valid_landmarks = results.pose_landmarks
                current_landmarks = results.pose_landmarks
                
                # Calculate knee angle for rep counting
                landmarks = results.pose_landmarks.landmark
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                current_knee_angle = knee_angle
                
                # Rep detection logic
                if not in_squat and knee_angle < MIN_KNEE_ANGLE:
                    # Just entered squat position
                    in_squat = True
                    current_rep_feedback = [] # Clear feedback for the new rep
                    
                elif in_squat:
                    # Check form while in squat
                    
                    # 1. Check for KNEE FALLING OVER TOE
                    left_knee_lm = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                    right_knee_lm = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    left_ankle_lm = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    right_ankle_lm = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    
                    knee_ankle_dist_left = left_knee_lm.x - left_ankle_lm.x
                    knee_ankle_dist_right = right_ankle_lm.x - right_knee_lm.x
                    
                    if knee_ankle_dist_left < -KNEE_ANKLE_DIST_THRESH or knee_ankle_dist_right < -KNEE_ANKLE_DIST_THRESH:
                        if "KNEE FALLING OVER TOE" not in current_rep_feedback:
                            current_rep_feedback.append("KNEE FALLING OVER TOE")
                            all_feedback.append("KNEE FALLING OVER TOE")
                    
                    # 2. Check for BEND BACKWARDS (based on hip angle)
                    # Calculate hip vertical angle
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    left_hip_lm = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    
                    # Calculate vertical reference
                    vertical_ref = [left_hip_lm.x, 0]  # Vertical line above hip
                    hip_vertical_angle = calculate_angle(
                        [left_shoulder.x, left_shoulder.y],
                        [left_hip_lm.x, left_hip_lm.y],
                        vertical_ref
                    )
                    
                    if hip_vertical_angle > HIP_THRESH_BACKWARDS:
                        if "BEND BACKWARDS" not in current_rep_feedback:
                            current_rep_feedback.append("BEND BACKWARDS")
                            all_feedback.append("BEND BACKWARDS")
                    
                    
                    # 4. Check for SQUAT TOO DEEP
                    if knee_angle < MAX_SQUAT_DEPTH:
                        if "SQUAT TOO DEEP" not in current_rep_feedback:
                            current_rep_feedback.append("SQUAT TOO DEEP")
                            all_feedback.append("SQUAT TOO DEEP")
                    
                    # Check if standing up
                    if knee_angle > MAX_KNEE_ANGLE:
                        # Just stood up from squat - Rep completed
                        in_squat = False
                        rep_just_completed = True
                        if not current_rep_feedback: # No issues detected during this rep
                            correct_rep_count += 1
                            completion_feedback = f"Correct Rep {correct_rep_count}"
                        else:
                            incorrect_rep_count += 1
                            # Combine issues for display
                            completion_feedback = f"Incorrect Rep {incorrect_rep_count}: {', '.join(current_rep_feedback[:2])}"
                        
                        # Add completion feedback to display list
                        active_feedback.append(completion_feedback)
                        feedback_frame_counters[completion_feedback] = 0
            
            elif last_valid_landmarks is not None:
                # Use last valid landmarks if no new ones detected
                current_landmarks = last_valid_landmarks
            
            # Draw landmarks on display frame
            if current_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    display_frame, 
                    current_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
            
            # Display knee angle if available
            if current_knee_angle is not None:
                angle_text = f"Knee: {int(current_knee_angle)}°"
                cv2.putText(display_frame, angle_text, (10, target_height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display rep counters
            correct_text = f"Correct: {correct_rep_count}"
            incorrect_text = f"Incorrect: {incorrect_rep_count}"
            cv2.putText(display_frame, correct_text, (target_width - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, incorrect_text, (target_width - 150, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Update feedback frame counters and remove expired feedback
            feedbacks_to_remove = []
            current_active_keys = list(feedback_frame_counters.keys())
            for feedback_key in current_active_keys:
                feedback_frame_counters[feedback_key] += 1
                if feedback_frame_counters[feedback_key] >= feedback_duration_frames:
                    feedbacks_to_remove.append(feedback_key)
            
            for feedback_key in feedbacks_to_remove:
                if feedback_key in active_feedback:
                    active_feedback.remove(feedback_key)
                if feedback_key in feedback_frame_counters:
                    del feedback_frame_counters[feedback_key]
            
            # Display active feedback with transparent background
            if active_feedback:
                # Create overlay for feedback text
                text_y_start = 50
                text_height = 35 * len(active_feedback) + 10
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (10, text_y_start - 10), 
                             (450, text_y_start + text_height), 
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
                
                # Add feedback text
                for i, feedback_text in enumerate(active_feedback):
                    y_pos = text_y_start + i * 35
                    cv2.putText(display_frame, feedback_text, (20, y_pos + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Enhance brightness if needed
            hsv = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            avg_brightness = np.mean(v)
            if avg_brightness < 100:
                lim = 255 - 30
                v[v > lim] = 255
                v[v <= lim] += 30
                hsv_adjusted = cv2.merge((h, s, v))
                display_frame = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
            
            # Write frame to output video
            out.write(display_frame)
            frames_written += 1
            frame_count += 1
            
            # Print progress periodically
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
    
    # Close video capture and writer
    cap.release()
    out.release()
    
    print(f"Video processing complete. Wrote {frames_written} frames. Correct: {correct_rep_count}, Incorrect: {incorrect_rep_count}")
    
    # Get unique feedback for summary
    unique_feedback = list(set(all_feedback))
    if not unique_feedback and correct_rep_count > 0:
        unique_feedback.append("Great form! No issues detected.")
    elif not unique_feedback and incorrect_rep_count == 0 and correct_rep_count == 0:
        unique_feedback.append("No reps detected or analyzed.")
    
    # Try to convert video with ffmpeg for browser compatibility
    try:
        compatible_output = os.path.join(output_dir, f"compatible_{video_filename}")
        cmd = ['ffmpeg', '-y', '-i', output_path, '-c:v', 'libx264', '-preset', 'fast', 
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', compatible_output]
        
        print(f"Running ffmpeg command: {' '.join(cmd)}")
        ffmpeg_result = subprocess.run(cmd, capture_output=True, text=True)
        
        if ffmpeg_result.returncode == 0 and os.path.exists(compatible_output):
            compat_size = os.path.getsize(compatible_output)
            print(f"FFmpeg conversion successful: {compatible_output}, size: {compat_size} bytes")
            if compat_size > 1000:
                os.replace(compatible_output, output_path)
                print(f"Replaced original with ffmpeg-converted version")
        else:
            print(f"FFmpeg error: {ffmpeg_result.stderr}")
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
    
    return {
        'feedback': unique_feedback, # Overall unique issues
        'video_path': video_filename,
        'correct_reps': correct_rep_count,
        'incorrect_reps': incorrect_rep_count
    }
