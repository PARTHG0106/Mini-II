from app import app, db
from models import WorkoutSchedule

def init_db():
    with app.app_context():
        # Create all tables
        db.create_all()
        print("Tables created successfully!")

if __name__ == "__main__":
    init_db() 