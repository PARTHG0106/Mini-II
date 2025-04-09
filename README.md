# Fit Me - AI-Powered Exercise Form Coach

Fit Me is a web application that uses AI to analyze exercise form in real-time, providing instant feedback and guidance to users during their workouts.

## Features

- Real-time exercise form analysis using AI
- Personalized workout schedules
- Exercise library with detailed instructions
- Progress tracking and analytics
- Video upload and analysis
- User authentication and profiles

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite
- **AI**: MediaPipe for pose detection
- **Deployment**: Local development server

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/PARTHG0106/Mini-II.git
cd Mini-II
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python init_db.py
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
Mini-II/
├── app.py                 # Main application file
├── init_db.py            # Database initialization script
├── models.py             # Database models
├── requirements.txt      # Python dependencies
├── static/              # Static files (CSS, JS, images)
│   ├── css/
│   └── js/
├── templates/           # HTML templates
│   ├── base.html
│   ├── landing.html
│   └── ...
└── uploads/            # User uploaded files
```

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for pose detection
- Flask for web framework
- Font Awesome for icons

Content
1. Sprints and Details
2. Images
3. Tests

# Planned Sprints
![image J9K5P2](https://github.com/ohksith/Workout-Form-Checker/assets/79146902/feb9f2e9-ee17-48b4-86ce-b5dee448a653)

# Week 1-4: 
## Sprint 1 - Front-End and Back-End Development
| Task | Status |
| ------------- | ------------- |
| Design: Designing System Architecture, Database Schema, API Schema.|X|
| Front-End: Build the basic structure using HTML, CSS, and JavaScript.|X|
| Back-End: Implement core functionalities with Flask.|X|
| Database: Set up PostgreSQL for handling user data.|X|
| Testing: Initial manual and automated tests on front-end and back-end to ensure basic functionality.|X|

Milestones: 
> Completion of front-end and back-end development.

# Week 5-7: 
## Sprint 2 - Model Training, Feedback Mechanism and Additional features
| Task | Status |
| ------------- | ------------- |
|Model Training: Train Media Pipe Model for joint detection and angle calculation.|X|
|Integration: Integrate Media Pipe and OpenCV for real-time video processing.|X|
|Feedback Mechanism: Implement logic for analyzing joint angles and providing real-time feedback.|X|
|Feedback Mechanism: Implement logic for post-workout feedback and tracking.|X|
|Additional Features: Alternate exercise options and personalized recommendations.|X|
|Testing: Manual and automated testing of the training and feedback mechanisms.|X|

Milestones:
> 1. Completion of model training and feedback mechanism.
> 2. Successful integration of Media Pipe and OpenCV.
> 3. Initial testing of the feedback mechanism.

   
# Week 8-10: 
## Sprint 3 - End-to-End Testing and Security Implementation
| Task | Status |
| ------------- | ------------- |
| End-to-End Testing: Conduct thorough end-to-end testing of the entire application.|X|
| Automated Testing: Implement Selenium for automated testing to ensure functionality, performance, and reliability.|X|
| Load Testing: Load testing for 100 concurrent users.|X|
| Security: Apply OWASP Top 10 security principles to protect user data and ensure application security.|X|
| Bug Fixing: Identify and fix any issues discovered during testing.|X|


Milestones:
>1. Comprehensive end-to-end testing completed.
>2. Automated testing with Selenium implemented.
>3. Security measures applied and tested.

 
# Week 11-13: 
## Sprint 4 - Deployment and Final Testing
| Task | Status |
| ------------- | ------------- |
| Deployment: Deploy the application on a cloud-based host.|x| - Amazon EC2 to the rescue
| User Testing: Conduct user testing in the deployed environment and gather feedback.|x|
| Final Testing: Perform final round of testing to ensure all features are working as expected.|x|


Milestones:
>1. Successful deployment of the application.
>2. Final round of testing and bug fixing completed.
>3. Project documentation finalized.

# Images
## UI - Some pictures of UI (Thank you DaisyUI, sorry performance nerds)
![image](https://github.com/user-attachments/assets/fbfcc492-e293-4b17-966f-cc6fb3a62f85)
![image](https://github.com/user-attachments/assets/02878d43-fccb-4379-81ff-ce1567d3e8b2)
![image](https://github.com/user-attachments/assets/7afc89f5-002a-431e-9117-807b9bfb9cca)
![image](https://github.com/user-attachments/assets/33f95f8a-5b48-4e6c-a41e-445a7a346f24)
![image](https://github.com/user-attachments/assets/9b82ac68-3d22-4b57-bbbf-30be91d7af03)

## Feedback
![image](https://github.com/user-attachments/assets/e904669e-f555-445f-9478-d24e1eb0efb4)
![image](https://github.com/user-attachments/assets/9a07538d-f329-416d-965c-9783bcaef0c9)
![image](https://github.com/user-attachments/assets/1fb81cbe-54b6-4648-9529-b869d8739b61)
![image](https://github.com/user-attachments/assets/1c2b900a-567b-426a-9968-53f064994410)

# Some Tests
![image](https://github.com/user-attachments/assets/b77a788a-a872-4626-9fc5-b74d68f05efc)
![image](https://github.com/user-attachments/assets/2caf8a60-ba96-4e65-88b5-9ffa664cf8e4)



