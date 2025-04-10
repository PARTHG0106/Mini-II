import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict
from models import db, UserExercise
import time
from flask import redirect, url_for
import os
import subprocess

live_feedback = ''

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
    tut_start_time = None
    total_tut_score = 0
    global live_feedback
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        return

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    max_angle = None
    min_angle = None
    prev_angle = None
    direction = None
    repetition_count = 0
    ex_info = defaultdict(dict)
    exercise_id = 3

    total_rom_score = 0

    start_time = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            print("Camera Fail")
            break

        elapsed_time = time.time() - start_time

        if elapsed_time < 5:
            remaining_time = 5 - int(elapsed_time)
            font_scale = 9
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 15

            text = str(remaining_time)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2

            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            cv2.putText(frame, 'Get into starting position!!!', (185, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if elapsed_time >= 5:
            break

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                points = [mp_pose.PoseLandmark.LEFT_HIP,
                          mp_pose.PoseLandmark.LEFT_KNEE,
                          mp_pose.PoseLandmark.LEFT_ANKLE]

                if points:
                    a = [landmarks[points[0].value].x, landmarks[points[0].value].y]
                    b = [landmarks[points[1].value].x, landmarks[points[1].value].y]
                    c = [landmarks[points[2].value].x, landmarks[points[2].value].y]
                    angle = calculate_angle(a, b, c)
                    color = (0, 255, 0) if 70 <= angle <= 180 else (0, 0, 255)

                    cv2.putText(image, str(int(angle)), tuple(np.multiply(b, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                    if angle < 80:
                        feedback = "Good form!"
                    elif angle < 100:
                        feedback = "Stand up more straight"
                    elif angle > 170:
                        feedback = "Good form!"
                    elif angle > 150:
                        feedback = "Bend more at the hips"
                    else:
                        feedback = ""

                    if feedback:
                        cv2.putText(image, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                    rom_score = 0

                    if prev_angle is not None:
                        if angle > 170 and direction != "up":
                            max_angle = angle
                            direction = "up"
                            tut_start_time = time.time()
                            print(f"Direction = UP. Max angle: {max_angle}")

                        elif angle < 90 and direction != "down":
                            if max_angle is not None:
                                repetition_count += 1
                                ex_info[repetition_count]['max'] = max_angle
                                ex_info[repetition_count]['min'] = angle

                                rep_rom_score = 0

                                if max_angle >= 170:
                                    rep_rom_score += 0.5
                                    print(f"Upper ROM: {rep_rom_score}")
                                else:
                                    print(f"Upper angle not reached: {max_angle}")

                                if angle <= 80:
                                    rep_rom_score += 0.5
                                    print(f"Lower ROM: {rep_rom_score}")
                                else:
                                    print(f"Lower angle not reached: {angle}")

                                total_rom_score += rep_rom_score
                                print(f"Rep ROM Score: {rep_rom_score}, Total ROM Score: {total_rom_score}")

                                if tut_start_time is not None:
                                    rep_tut_score = time.time() - tut_start_time
                                    total_tut_score += rep_tut_score
                                    print(f"Rep {repetition_count} TUT: {rep_tut_score:.2f} seconds")
                                    tut_start_time = None

                                max_angle = None
                                direction = "down"
                                print("Direction reset to DOWN.")
                            else:
                                print("max_angle is None")

                    prev_angle = angle


                    cv2.putText(image, f'Rep {repetition_count}', (30, 150),
                                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

                    if repetition_count >= rep_goal:
                        end_time = time.time() + 5

                        while time.time() < end_time:
                            success, frame = camera.read()
                            if not success:
                                print("Failed to read frame from camera.")
                                break

                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image.flags.writeable = False
                            results = pose.process(image)
                            image.flags.writeable = True
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            cv2.putText(image, 'Great Job! '
                                               'Click the feedback button after page '
                                               'stops loading', (30, 200),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 0, 255), 1, cv2.LINE_AA)

                            ret, buffer = cv2.imencode('.jpg', image)
                            frame = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                        try:
                            from app import app

                            with app.app_context():
                                print(user_id, rom_score)
                                new_exercise = UserExercise(
                                    user_id=user_id,
                                    exercise_id=exercise_id,
                                    total_reps=rep_goal,
                                    rom_score=total_rom_score,
                                    tut_score=total_tut_score,
                                    count=rep_goal
                                )
                                db.session.add(new_exercise)
                                db.session.commit()
                                print('Data transferred!')

                                ex_info.clear()
                                repetition_count = 0
                                redirect_url = '/dash/'
                                return f'<html><head><meta http-equiv="refresh" content="0; url={redirect_url}" /></head><body></body></html>'
                        except Exception as e:
                            print(f"Error while appending to db: {e}")
                        finally:
                            break

            except Exception as e:
                print(f"Error during pose processing: {e}")

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

def analyze_deadlift_video(video_path):
    """
    Analyzes deadlift video and provides feedback on form
    """
    print(f"Starting analysis of video: {video_path}")
    mp_pose = mp.solutions.pose
    
    # Path for processed video
    timestamp = int(time.time())
    video_filename = f"deadlift_analysis_{timestamp}.mp4"
    
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
    in_deadlift = False  # Track if currently in deadlift position
    current_rep_feedback = [] # Feedback generated during the current rep
    
    # Rep detection thresholds
    MIN_HIP_ANGLE = 30  # Below this is considered a deadlift
    MAX_HIP_ANGLE = 160  # Above this is considered fully extended
    
    # Thresholds for specific feedback
    BACK_ANGLE_THRESH = 30  # Above this angle indicates improper back position
    KNEE_ANGLE_THRESH = 45  # Below this angle indicates improper knee position
    
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
            current_hip_angle = None
            rep_just_completed = False
            
            # Check if we have valid landmarks
            if results.pose_landmarks:
                last_valid_landmarks = results.pose_landmarks
                current_landmarks = results.pose_landmarks
                
                # Calculate hip angle for rep counting
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                
                hip_angle = calculate_angle(shoulder, hip, knee)
                current_hip_angle = hip_angle
                
                # Rep detection logic
                if not in_deadlift and hip_angle < MIN_HIP_ANGLE:
                    # Just entered deadlift position
                    in_deadlift = True
                    current_rep_feedback = [] # Clear feedback for the new rep
                    
                elif in_deadlift:
                    # Check form while in deadlift
                    
                    # 1. Check for IMPROPER BACK POSITION
                    back_angle = calculate_angle(
                        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    )
                    
                    if back_angle > BACK_ANGLE_THRESH:
                        if "IMPROPER BACK POSITION" not in current_rep_feedback:
                            current_rep_feedback.append("IMPROPER BACK POSITION")
                            all_feedback.append("IMPROPER BACK POSITION")
                    
                    # 2. Check for IMPROPER KNEE POSITION
                    if hip_angle < KNEE_ANGLE_THRESH:
                        if "IMPROPER KNEE POSITION" not in current_rep_feedback:
                            current_rep_feedback.append("IMPROPER KNEE POSITION")
                            all_feedback.append("IMPROPER KNEE POSITION")
                    
                    # Check if fully extended
                    if hip_angle > MAX_HIP_ANGLE:
                        # Just completed deadlift - Rep completed
                        in_deadlift = False
                        rep_just_completed = True
                        if not current_rep_feedback: # No issues detected during this rep
                            correct_rep_count += 1
                            completion_feedback = f"Good Rep {correct_rep_count}! Keep it up!"
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
            
            # Display hip angle if available
            if current_hip_angle is not None:
                angle_text = f"Hip: {int(current_hip_angle)}°"
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
