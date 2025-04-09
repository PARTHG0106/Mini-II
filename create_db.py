from app import app, db

with app.app_context():
    db.drop_all()  # This will drop all existing tables
    db.create_all()  # This will create all tables defined in your models
    print("Database tables created successfully!") 