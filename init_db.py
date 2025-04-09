from app import app, db
from models import User, Exercises, UserExercise, WorkoutSchedule, ScheduleCompletion

def init_db():
    with app.app_context():
        # Drop all existing tables
        db.drop_all()
        
        # Create all tables
        db.create_all()
        
        # Create default user
        default_user = User(
            username='demo',
            email='demo@example.com',
            rep_goal=8,
            ex_goal=5
        )
        default_user.set_password('demo123')
        db.session.add(default_user)
        
        # Add basic exercises
        exercises = [
            Exercises(name='Shoulder Press', link='https://www.youtube.com/embed/qEwKCR5JCog', muscles_involved='Shoulders,Triceps'),
            Exercises(name='Bicep Curls', link='https://www.youtube.com/embed/ykJmrZ5v0Oo', muscles_involved='Biceps,Forearms'),
            Exercises(name='Barbell Squats', link='https://www.youtube.com/embed/SW_C1A-rejs', muscles_involved='Quadriceps,Hamstrings,Glutes'),
            Exercises(name='Deadlift', link='https://www.youtube.com/embed/op9kVnSso6Q', muscles_involved='Back,Legs,Core'),
            Exercises(name='Lateral Raises', link='https://www.youtube.com/embed/3VcKaXpzqRo', muscles_involved='Shoulders'),
            Exercises(name='Push-ups', link='https://www.youtube.com/embed/IODxDxX7oi4', muscles_involved='Chest,Shoulders,Triceps'),
            Exercises(name='Pull-ups', link='https://www.youtube.com/embed/eGo4IYlbE5g', muscles_involved='Back,Biceps'),
            Exercises(name='Plank', link='https://www.youtube.com/embed/pSHjTRCQxIw', muscles_involved='Core'),
            Exercises(name='Crunches', link='https://www.youtube.com/embed/Xyd_fa5zoEU', muscles_involved='Core'),
            Exercises(name='Lunges', link='https://www.youtube.com/embed/3XDriUn0udo', muscles_involved='Quadriceps,Hamstrings,Glutes'),
            Exercises(name='Russian Twists', link='https://www.youtube.com/embed/wkD8rjkodUI', muscles_involved='Core,Obliques')
        ]
        
        for exercise in exercises:
            db.session.add(exercise)
        
        db.session.commit()
        print("Database initialized successfully with default user and basic exercises!")

if __name__ == "__main__":
    init_db() 