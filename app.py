from flask import Flask, flash, render_template, redirect, request, url_for, jsonify, session, Response, send_file
from forms import LoginForm, SearchForm, RegistrationForm
from flask_migrate import Migrate
from config import Config
from models import User, db, bcrypt, Exercises, UserExercise, ExerciseUpload, WorkoutSchedule, ScheduleCompletion
from shoulder_press import gen_frames as gen_frames_shoulder_press, analyze_shoulder_press_video
from bicep_curls import gen_frames as gen_frames_bicep_curls, analyze_bicep_curls_video
from barbell_squats import gen_frames as gen_frames_barbell_squats, analyze_squat_video
from deadlift import gen_frames as gen_frames_deadlift, analyze_deadlift_video
from lateral_raises import gen_frames as gen_frames_lateral_raises, analyze_lateral_raise_video
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from sqlalchemy import func
from functools import wraps
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import mediapipe as mp
import time

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
bcrypt.init_app(app)
migrate = Migrate(app, db)
quartz = dbc.themes.SKETCHY

# Configure upload folder
UPLOAD_FOLDER = 'uploads/exercises'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Decorator to ensure user is logged in before accessing certain routes
def login_required(f):
    @wraps(f)
    def chck(*args, **kwargs):
        if 'user_id' not in session:
            flash("You need to log in first.", "danger")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return chck

# Dash application for interactive data visualisation
dash_app = Dash(__name__, server=app, external_stylesheets=[quartz], url_base_pathname='/dashboard/')

# Set up layout for the Dash application,
dash_app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.A("Go Back", href="/mainboard", className="btn btn-lg text-center text-white", style={
                "background-color": "#98ff98",  # Green mint
                "border-radius": "10px",
                "padding": "10px 20px",
                "display": "block",
                "margin": "0 auto",
                "text-decoration": "none",
                "font-size": "24px",
                "width": "200px",
                "box-shadow": "2px 2px 5px rgba(0, 0, 0, 0.3)"
            }),
            width=12
        )
    ),
    dcc.Tabs(id="tabs-example", value='tab-1', children=[
        dcc.Tab(label='Last Workout', value='tab-1'),
        dcc.Tab(label='Progress', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

# Callback to dynamically update content based on selected tabs
@dash_app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-example', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return dbc.Container([
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='exercise-dropdown',
                        options=[
                            {'label': 'Shoulder Press', 'value': 1},#edit
                            {'label': 'Bicep Curl', 'value': 2},
                            {'label': 'Barbell Squats', 'value': 3},
                            {'label': 'Deadlift', 'value': 4},
                            {'label': 'Lateral Raises', 'value': 5}

                        ],
                        value=1,
                        className='mx-auto',
                        style={'width': '50%'}
                    ),
                    width=6
                )
            ], justify="center"),
            html.Div(id='exercise-output', className="mt-4")
        ])
    elif tab == 'tab-2':
        return dbc.Container([
            dbc.Row(
                dbc.Col(html.H3("Overall Progress", className="text-center my-4"), width=12)
            ),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='overall-progress-graph'),
                    width=12
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='specific-exercise-dropdown',
                        options=[
                            {'label': 'Shoulder Press', 'value': 1},
                            {'label': 'Bicep Curl', 'value': 2},
                            {'label': 'Barbell Squats', 'value': 3},
                            {'label': 'Deadlift', 'value': 4},
                            {'label': 'Lateral Raises', 'value': 5}
                        ],
                        value=1,
                        className='mx-auto',
                        style={'width': '50%'}
                    ),
                    width=6
                )
            ], justify="center", className="mt-4"),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='specific-exercise-progress-graph'),
                    width=12
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='muscles-hit-graph'),  # Later
                    width=12
                )
            ])
        ])


# Callback for the Last Workout Tab
@dash_app.callback(
    Output('exercise-output', 'children'),
    Input('exercise-dropdown', 'value')
)
def update_last_workout(exercise_id):
    last_workout = UserExercise.query.filter_by(user_id=session['user_id'], exercise_id=exercise_id).order_by(
        UserExercise.date.desc()).first()

    if last_workout:
        workout_data = {
            'Date': last_workout.date.strftime("%Y-%m-%d %H:%M:%S"),
            'ROM Score': last_workout.rom_score,
            'TUT Score': round(last_workout.tut_score / last_workout.total_reps, 1),
            'Total Reps': last_workout.total_reps,
            'rom_score': last_workout.rom_score,
            'Count': last_workout.count
        }

        pie_chart = go.Figure(data=[go.Pie(labels=['Efficient Reps', 'Missed Reps'],
                                           values=[workout_data['Total Reps'],
                                                   workout_data['Total Reps'] - workout_data['rom_score']])])
        pie_chart.update_layout(
            title={
                'text': 'Efficiency in Last Workout',
                'font': {
                    'color': 'black'
                }
            },
            legend={
                'font': {
                    'color': 'black'
                }
            },
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return html.Div([
            html.H4(f"Last Workout: {workout_data['Date']}"),
            html.P(f"ROM Score: {workout_data['ROM Score']}"),
            html.P(f"TUT: {workout_data['TUT Score']} sec per rep"),
            html.P(f"Total Reps: {workout_data['Total Reps']}"),
            dcc.Graph(figure=pie_chart)
        ])
    else:
        return html.P("No workout data available.")


# Callback to the Progress Tab
@dash_app.callback(
    [Output('overall-progress-graph', 'figure'),
     Output('specific-exercise-progress-graph', 'figure'),
     Output('muscles-hit-graph', 'figure')],
    [Input('tabs-example', 'value'),
     Input('specific-exercise-dropdown', 'value')]
)
def update_progress(tab, exercise_id):
    if tab != 'tab-2':
        raise PreventUpdate

    # Overall Progress DataFrame
    overall_data = UserExercise.query.filter_by(user_id=session['user_id']).all()
    overall_df = pd.DataFrame([{
        'date': record.date,
        'rom_score': record.rom_score,
        'tut_score': record.tut_score,
        'rep_number': record.total_reps,
        'count': record.count,
    } for record in overall_data])

    if overall_df.empty:
        return go.Figure(), go.Figure(), go.Figure()

    overall_df['week'] = overall_df['date'].dt.strftime('%Y-%U')
    overall_df['efficiency'] = (overall_df['count'] / overall_df['rep_number']) * 100
    weekly_efficiency_df = overall_df.groupby('week').agg({'efficiency': 'mean'}).reset_index()
    weekly_efficiency_df['wow_improvement'] = weekly_efficiency_df['efficiency'].pct_change() * 100
    wow = f'{weekly_efficiency_df["wow_improvement"].iloc[-1]:.2f}% better than prev. week ⬆️'

    overall_df = overall_df.groupby('date').agg({
        'rom_score': 'mean',
        'tut_score': 'mean',
        'rep_number': 'sum',
        'count': 'sum'
    }).reset_index()

    overall_line_chart = go.Figure()
    overall_line_chart.add_trace(go.Scatter(x=overall_df['date'], y=overall_df['count'],
                                            mode='lines+markers', name='Efficient Reps'))
    overall_line_chart.add_trace(go.Scatter(x=overall_df['date'], y=overall_df['rep_number'],
                                            mode='lines+markers', name='Total Reps'))
    overall_line_chart.add_annotation(
        text=f"WoW Improvement: {wow}",
        xref="paper", yref="paper",
        x=0.5, y=1.1,
        showarrow=False,
        font=dict(
            size=14,
            color="black"
        ),
        align="center",
        bgcolor="white",
        opacity=0.8
    )
    overall_line_chart.update_layout(title='Overall Week-Over-Week Progress',
                                     xaxis_title='Date', yaxis_title='Efficiency')


    specific_data = UserExercise.query.filter_by(exercise_id=exercise_id, user_id=session['user_id']).all()
    specific_df = pd.DataFrame([{
        'date': record.date,
        'rom_score': record.rom_score,
        'tut_score': record.tut_score,
        'rep_number': record.total_reps,
        'count': record.count,
    } for record in specific_data])

    specific_line_chart = go.Figure()
    if not specific_df.empty:
        specific_df = specific_df.groupby('date').agg({
            'rom_score': 'mean',
            'tut_score': 'mean',
            'rep_number': 'sum',
            'count': 'sum'
        }).reset_index()

        specific_line_chart.add_trace(go.Scatter(x=specific_df['date'], y=specific_df['count'],
                                                 mode='lines+markers', name='Efficient Reps'))
        specific_line_chart.add_trace(go.Scatter(x=specific_df['date'], y=specific_df['rep_number'],
                                                 mode='lines+markers', name='Total Reps'))
        specific_line_chart.update_layout(title='Specific Exercise Progress',
                                          xaxis_title='Date', yaxis_title='Efficiency')


    muscle_data = db.session.query(Exercises.muscles_involved, db.func.sum(UserExercise.total_reps)).join(
        UserExercise, Exercises.id == UserExercise.exercise_id).filter(UserExercise.user_id == session['user_id']).group_by(
        Exercises.muscles_involved).all()

    muscle_df = pd.DataFrame(muscle_data, columns=['Muscles Involved', 'Total Reps'])
    muscle_dict = {}

    for muscles, reps in zip(muscle_df['Muscles Involved'], muscle_df['Total Reps']):

        muscle_list = muscles.split(',')

        for muscle in muscle_list:
            muscle = muscle.strip()
            if muscle in muscle_dict:
                muscle_dict[muscle] += reps
            else:
                muscle_dict[muscle] = reps

    muscle_ind_df = pd.DataFrame(list(muscle_dict.items()), columns=['muscle', 'Total Reps'] )
    muscle_ind_df = muscle_ind_df.sort_values(by='Total Reps', ascending=False)
    muscle_bar_chart = go.Figure(data=[
        go.Bar(x=muscle_ind_df['muscle'], y=muscle_ind_df['Total Reps'])
    ])
    muscle_bar_chart.update_layout(title='Muscles Worked', xaxis_title='Muscle Groups', yaxis_title='Total Reps')

    return overall_line_chart, specific_line_chart, muscle_bar_chart


# Flask route to render the Dash dashboard
@app.route("/dash/")
@login_required
def render_dashboard():
    if 'user_id' not in session:
        flash('Please login to access the dashboard', 'error')
        return redirect(url_for('login'))
    return dash_app.index()

# Function to generate frames for real-time video feedback
def gen_frames(exercise, user_id, rep_goal):
    if exercise == 'shoulder_press':
        print('s_press')
        return gen_frames_shoulder_press(user_id, rep_goal)
    elif exercise == 'dumbbell_curls':
        print('b_curls')
        return gen_frames_bicep_curls(user_id, rep_goal)
    elif exercise == 'barbell_squats':
        print('b_squats')
        return gen_frames_barbell_squats(user_id, rep_goal)
    elif exercise == 'deadlift':
        print('dlift')
        return gen_frames_deadlift(user_id, rep_goal)
    elif exercise == 'lateral_raises':
        print('l_raises')
        return gen_frames_lateral_raises(user_id, rep_goal)
    else:
        return None


# Flask routes for user authentication
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        login_form = LoginForm()
        registration_form = RegistrationForm()

        if request.method == 'POST':
            print("POST request received")
            print("Form data:", request.form)
            
            # Check which form was submitted based on the submit button name
            if 'register' in request.form:
                print("Registration form submitted")
                if registration_form.validate_on_submit():
                    print("Registration form validated")
                    # Check if username already exists
                    user = User.query.filter_by(username=registration_form.username.data).first()
                    if user is None:
                        print("Creating new user")
                        new_user = User(
                            username=registration_form.username.data,
                            email=registration_form.email.data
                        )
                        new_user.set_password(registration_form.password.data)
                        db.session.add(new_user)
                        db.session.commit()
                        print("New user created")
                        flash('Registration successful! You can now log in.', 'success')
                        return redirect(url_for('login'))
                    else:
                        print("Username already exists")
                        flash('Username already exists. Please choose a different username.', 'danger')
                else:
                    print("Registration validation failed:", registration_form.errors)
                    for field, errors in registration_form.errors.items():
                        for error in errors:
                            flash(f'{field}: {error}', 'danger')
            
            else:  # Login form submitted
                print("Login form submitted")
                if login_form.validate_on_submit():
                    print("Login form validated")
                    user = User.query.filter_by(username=login_form.username.data).first()
                    print("User found:", user is not None)
                    if user and user.check_password(login_form.password.data):
                        print("Password verified")
                        flash('Login successful!', 'success')
                        session['user_id'] = user.id
                        return redirect(url_for('mainboard'))
                    else:
                        print("Invalid credentials")
                        flash('Invalid username or password.', 'danger')
                else:
                    print("Login validation failed:", login_form.errors)
                    for field, errors in login_form.errors.items():
                        for error in errors:
                            flash(f'{field}: {error}', 'danger')

        return render_template('enter.html', login_form=login_form, registration_form=registration_form)
    except Exception as e:
        print("Error in login route:", str(e))
        import traceback
        print("Traceback:", traceback.format_exc())
        flash('An error occurred. Please try again.', 'danger')
        return render_template('enter.html', login_form=LoginForm(), registration_form=RegistrationForm())


from datetime import datetime, timedelta

# Flask route for homepage
@app.route('/mainboard', methods=['GET', 'POST'])
@login_required
def mainboard():
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found. Please log in again.', 'error')
        return redirect(url_for('login'))
        
    schedules = WorkoutSchedule.query.filter_by(user_id=session['user_id']).all()
    now = datetime.now()

    # Calculate workout efficiency
    total_exercises = UserExercise.query.filter_by(user_id=session['user_id']).count()
    if total_exercises > 0:
        efficiency = db.session.query(
            func.sum(UserExercise.count) * 100 / func.sum(UserExercise.total_reps)
        ).filter(UserExercise.user_id == session['user_id']).scalar() or 0
    else:
        efficiency = 0

    # Calculate total reps
    total_reps = db.session.query(
        func.sum(UserExercise.count)
    ).filter(UserExercise.user_id == session['user_id']).scalar() or 0

    # Calculate exercises this week
    week_start = now - timedelta(days=now.weekday())
    exercises_this_week = UserExercise.query.filter(
        UserExercise.user_id == session['user_id'],
        UserExercise.date >= week_start
    ).count()

    # Calculate workout streak
    workout_days = db.session.query(
        func.count(func.distinct(func.date(UserExercise.date)))
    ).filter(
        UserExercise.user_id == session['user_id'],
        UserExercise.date >= now - timedelta(days=30)
    ).scalar() or 0

    # Get most performed exercise
    most_performed = db.session.query(
        Exercises.name,
        func.count(UserExercise.id).label('count')
    ).join(UserExercise, UserExercise.exercise_id == Exercises.id).filter(
        UserExercise.user_id == session['user_id']
    ).group_by(Exercises.name).order_by(func.count(UserExercise.id).desc()).first()

    selected_exercise_name = most_performed[0] if most_performed else "No exercises yet"

    # Get alternative exercise suggestion
    if most_performed:
        alternate = Exercises.query.filter(
            Exercises.muscles_involved.like(f"%{most_performed[0]}%"),
            Exercises.name != most_performed[0]
        ).first()
        alternate_message = alternate.name if alternate else "Try a new exercise!"
    else:
        alternate_message = "Start with any exercise!"

    # Calculate streak
    streak = 0
    current_date = now.date()
    while True:
        has_workout = UserExercise.query.filter(
            UserExercise.user_id == session['user_id'],
            func.date(UserExercise.date) == current_date
        ).first() is not None
        
        if not has_workout:
            break

        streak += 1
        current_date -= timedelta(days=1)

    return render_template('mainboard.html', 
                         user=user,
                         schedules=schedules,
                         now=now,
                         efficiency=round(efficiency),
                         total_exercises=total_exercises,
                         exercises_this_week=exercises_this_week,
                         ex_goal=user.ex_goal,
                         streak=streak,
                         workout_days=workout_days,
                         total_reps=total_reps,
                         selected_exercise_name=selected_exercise_name,
                         alternate_message=alternate_message)



# Route for profile page
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    search_form = SearchForm()
    user_id = session.get('user_id')

    user_details = User.query.filter_by(id=user_id).first()

    if user_details:
        username = user_details.username
        ex_goal = user_details.ex_goal
        rep_goal = user_details.rep_goal

    if request.method == 'POST':
        if 'change_password' in request.form:
            new_password = request.form['new_password']
            user_details.set_password(new_password)
            db.session.commit()
            flash('Password updated successfully!', 'success')

        elif 'increase_ex_goal' in request.form:
            user_details.ex_goal += 1
            db.session.commit()

        elif 'decrease_ex_goal' in request.form:
            if user_details.ex_goal > 0:
                user_details.ex_goal -= 1
                db.session.commit()

        elif 'increase_rep_goal' in request.form:
            user_details.rep_goal += 1
            db.session.commit()

        elif 'decrease_rep_goal' in request.form:
            if user_details.rep_goal > 0:
                user_details.rep_goal -= 1
                db.session.commit()

        elif 'delete_account' in request.form:
            db.session.delete(user_details)
            db.session.commit()
            flash('Account deleted successfully!', 'success')
            return redirect(url_for('logout'))

    return render_template('profile.html', search_form=search_form,
                           username=username, ex_goal=ex_goal, rep_goal=rep_goal)

#
'''@app.route('/search_exercises', methods=['GET'])
@login_required
def search_exercises():
    query = request.args.get('query', '').strip()
    if query:
        exercises = Exercises.query.filter(Exercises.name.ilike(f'%{query}%')).all()
        results = [{'name': exercise.name, 'link': exercise.link} for exercise in exercises]
        return jsonify(results)'''

# Route to learn exercise
@app.route('/exercises', methods=['GET', 'POST'])
@login_required
def exercises():
    exercises = Exercises.query.all()
    schedules = WorkoutSchedule.query.filter_by(user_id=session['user_id']).all()
    now = datetime.now()
    
    # Get the selected exercise from the URL parameter
    selected_exercise = request.args.get('exercise')
    video_link = None
    
    # If an exercise is selected, get its YouTube link
    if selected_exercise:
        exercise = Exercises.query.filter_by(name=selected_exercise).first()
        if exercise:
            video_link = exercise.link
    
    return render_template('exercises.html', 
                          exercises=exercises, 
                          schedules=schedules, 
                          now=now,
                          video_link=video_link)

# Route for leaderboard
@app.route('/leaderboard', methods=['GET', 'POST'])
@login_required
def leaderboard():
    search_form = SearchForm()
    view = request.form.get('view', 'total_exercises')


    highest_exercises = db.session.query(
        User.id,
        User.username,
        func.count(UserExercise.id).label('exercise_count')
    ).join(User, User.id == UserExercise.user_id).group_by(User.id).order_by(func.count(UserExercise.id).desc()).all()


    highest_reps = db.session.query(
        User.id,
        User.username,
        func.sum(UserExercise.count).label('total_reps')
    ).join(User, User.id == UserExercise.user_id).group_by(User.id).order_by(func.sum(UserExercise.count).desc()).all()

    return render_template('leaderboard.html', leaderboard_data=highest_exercises if view == 'total_exercises'
    else highest_reps, view=view, search_form=search_form)


@app.route('/workout')
@login_required
def workout():
    search_form = SearchForm()
    return render_template('exercises.html', search_form=search_form)

# Routes to start exercise
# Temp page to re-direct to exercise page
@app.route('/start/<exercise>')
@login_required
def start(exercise):
    search_form = SearchForm()
    user_id = session.get('user_id')
    rep_goal = db.session.query(User.rep_goal).filter_by(id=user_id).scalar()

    return render_template('instructions.html', search_form=search_form, exercise=exercise, user_id=user_id, rep_goal=rep_goal)

# Actual Exercise Page
@app.route('/start_page/<exercise>')
@login_required
def start_page(exercise):
    search_form = SearchForm()
    user_id = session.get('user_id')
    rep_goal = db.session.query(User.rep_goal).filter_by(id=user_id).scalar()



    video_feed_url = url_for('video_feed', exercise=exercise, user_id=user_id, rep_goal=rep_goal)

    return render_template('start.html', search_form=search_form, exercise=exercise, user_id=user_id, rep_goal=rep_goal, video_feed_url=video_feed_url)

# Video feed linked to start.html
@app.route('/video_feed/<exercise>/<int:user_id>/<int:rep_goal>')
def video_feed(exercise, user_id, rep_goal):
    return Response(gen_frames(exercise, user_id, rep_goal), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to logout
@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/upload_exercise', methods=['GET', 'POST'])
@login_required
def upload_exercise():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file part', 'error')
            return redirect(request.url)
        
        video_file = request.files['video']
        if video_file.filename == '':
            flash('No selected video file', 'error')
            return redirect(request.url)
        
        if not allowed_file(video_file.filename):
            flash('Invalid file format. Supported formats: mp4, mov, avi', 'error')
            return redirect(request.url)
        
        exercise_type = request.form.get('exercise_type')
        notes = request.form.get('notes', '')
        
        # Save uploaded video
        video_filename = f"{exercise_type}_{int(time.time())}.mp4"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        video_file.save(video_path)
        
        # Analyze the video based on exercise type
        analysis_result = None
        if exercise_type == 'bicep_curl':
            analysis_result = analyze_bicep_curls_video(video_path)
        elif exercise_type == 'squat':
            analysis_result = analyze_squat_video(video_path)
        elif exercise_type == 'shoulder_press':
            analysis_result = analyze_shoulder_press_video(video_path)
        elif exercise_type == 'deadlift':
            analysis_result = analyze_deadlift_video(video_path)
        elif exercise_type == 'lateral_raise':
            analysis_result = analyze_lateral_raise_video(video_path)
            
        if not analysis_result or 'error' in analysis_result:
            flash('Error analyzing video: ' + analysis_result.get('error', 'Unknown error'), 'error')
            return redirect(url_for('upload_exercise'))
            
        # Get the analyzed video path from the result
        analyzed_filename = analysis_result.get('video_path', f"analyzed_{video_filename}")
            
        # Save exercise data
        exercise = ExerciseUpload(
            user_id=session['user_id'],
            exercise_type=exercise_type,
            video_path=analyzed_filename,
            feedback='\n'.join(analysis_result.get('feedback', [])),
            notes=notes
        )
        db.session.add(exercise)
        db.session.commit()
        
        flash('Exercise video uploaded and analyzed successfully!', 'success')
        return redirect(url_for('upload_exercise'))
        
    # Get user's analyzed videos
    analyzed_videos = ExerciseUpload.query.filter_by(user_id=session['user_id']).order_by(ExerciseUpload.created_at.desc()).all()
    return render_template('upload_exercise.html', analyzed_videos=analyzed_videos)

@app.route('/delete-analyzed-video/<int:upload_id>', methods=['POST'])
@login_required
def delete_analyzed_video(upload_id):
    upload = ExerciseUpload.query.get_or_404(upload_id)
    
    # Ensure the user can only delete their own videos
    if upload.user_id != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('upload_exercise'))
    
    try:
        # Delete the video file
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], upload.video_path)
        if os.path.exists(video_path):
            os.remove(video_path)
        
        # Delete the database record
        db.session.delete(upload)
        db.session.commit()
        
        flash('Video deleted successfully!', 'success')
    except Exception as e:
        print(f"Error deleting video: {str(e)}")
        db.session.rollback()
        flash('Error deleting video', 'error')
    
    return redirect(url_for('upload_exercise'))

@app.route('/schedule', methods=['GET'])
@login_required
def schedule():
    exercises = Exercises.query.all()
    schedules = WorkoutSchedule.query.filter_by(user_id=session['user_id']).all()
    now = datetime.now()
    return render_template('schedule.html', exercises=exercises, schedules=schedules, now=now)

@app.route('/add_schedule', methods=['POST'])
@login_required
def add_schedule():
    exercise_id = request.form.get('exercise_id')
    day_of_week = int(request.form.get('day_of_week'))
    sets = int(request.form.get('sets'))
    reps = int(request.form.get('reps'))
    
    new_schedule = WorkoutSchedule(
        user_id=session['user_id'],
        exercise_id=exercise_id,
        day_of_week=day_of_week,
        sets=sets,
        reps=reps
    )
    
    db.session.add(new_schedule)
    db.session.commit()
    
    flash('Exercise added to schedule successfully!', 'success')
    return redirect(url_for('schedule'))

@app.route('/delete_schedule/<int:schedule_id>', methods=['POST'])
@login_required
def delete_schedule(schedule_id):
    schedule = WorkoutSchedule.query.get_or_404(schedule_id)
    
    if schedule.user_id != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('schedule'))
    
    db.session.delete(schedule)
    db.session.commit()
    
    flash('Exercise removed from schedule successfully!', 'success')
    return redirect(url_for('schedule'))

@app.route('/update_schedule_status/<int:schedule_id>', methods=['POST'])
@login_required
def update_schedule_status(schedule_id):
    try:
        data = request.get_json()
        schedule = WorkoutSchedule.query.get_or_404(schedule_id)
        
        # Verify user owns this schedule
        if schedule.user_id != session['user_id']:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403

        # Get the current week's start date
        current_date = datetime.now()
        week_start = current_date - timedelta(days=current_date.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        # Check if there's a completion record for this schedule in the current week
        completion = ScheduleCompletion.query.filter(
            ScheduleCompletion.schedule_id == schedule_id,
            ScheduleCompletion.completed_at >= week_start
        ).first()

        if data['is_completed']:
            if not completion:
                # Create new completion record for this week
                completion = ScheduleCompletion(
                    schedule_id=schedule_id,
                    completed_at=current_date
                )
                db.session.add(completion)
        else:
            if completion:
                # Remove completion record for this week
                db.session.delete(completion)
        
        db.session.commit()
        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

# Route to get schedule completion status
@app.route('/get_schedule_status/<int:schedule_id>')
@login_required
def get_schedule_status(schedule_id):
    try:
        schedule = WorkoutSchedule.query.get_or_404(schedule_id)
        
        # Verify user owns this schedule
        if schedule.user_id != session['user_id']:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403

        # Get the current week's start date
        current_date = datetime.now()
        week_start = current_date - timedelta(days=current_date.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        # Check if there's a completion record for this week
        completion = ScheduleCompletion.query.filter(
            ScheduleCompletion.schedule_id == schedule_id,
            ScheduleCompletion.completed_at >= week_start
        ).first()

        return jsonify({
            'success': True,
            'is_completed': completion is not None
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/download-analyzed-video/<int:upload_id>')
@login_required
def download_analyzed_video(upload_id):
    upload = ExerciseUpload.query.get_or_404(upload_id)
    
    # Ensure the user can only download their own videos
    if upload.user_id != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('upload_exercise'))
    
    try:
        # Full path to the video file
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], upload.video_path)
        
        # Check if file exists
        if not os.path.exists(video_path):
            flash('Video file not found', 'error')
            return redirect(url_for('upload_exercise'))
        
        # Check if download parameter is provided
        download = request.args.get('download', 'false') == 'true'
        
        # Determine file size
        file_size = os.path.getsize(video_path)
        
        response = send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=download,
            download_name=f"{upload.exercise_type}_analyzed.mp4" if download else None
        )
        
        # Add Content-Length header
        response.headers['Content-Length'] = file_size
        
        return response
    except Exception as e:
        print(f"Error serving video: {str(e)}")
        flash('Error serving video', 'error')
        return redirect(url_for('upload_exercise'))

if __name__ == "__main__":
    app.debug = True  # Enable debug mode
    app.run()
