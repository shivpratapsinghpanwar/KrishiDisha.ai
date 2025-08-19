import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from flask import Flask, render_template, request, redirect, url_for, flash, session,send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from reportlab.lib.pagesizes import letter

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
import joblib

from PIL import Image as PILImage

import torchvision.transforms.functional as TF
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from io import BytesIO
import imghdr
import os
import torch
import CNN

# Initialize Flask app
app = Flask(__name__,static_folder='static')


base = r'D:\KrishiDisha\Crop and Fertilizer Recommendation - Copy'

# Defining the paths with base directory and filenames
disease_info_path = os.path.join(base, 'data', 'disease_info.csv')
supplement_info_path = os.path.join(base, 'data','supplement_info.csv')
data_path = os.path.join(base, 'data','crop_yield.csv')

model_path = os.path.join(base, 'models','plant_disease_model_1_latest.pt')
pipeline_path = (r'Crop and Fertilizer Recommendation - Copy\models\pipeline_yield_prediction_pipeline.pkl')
# Load Data
disease_info = pd.read_csv(disease_info_path, encoding='cp1252')
supplement_info = pd.read_csv(supplement_info_path, encoding='cp1252')

# Load Model
model_disease = CNN.CNN(39)
model_disease.load_state_dict(torch.load(model_path))
model_disease.eval()

# Create Uploads Folder if not exists
upload_folder = os.path.join('static', 'uploads')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

def prediction(image_path):
    image = PILImage.open(image_path)
    image = image.resize((224, 224))  # Resize image to the expected size
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))  # Reshape for the model
    output = model_disease(input_data)
    output = output.detach().numpy()  # Convert to numpy array
    index = np.argmax(output)  # Get the index with the highest probability
    return index



model = joblib.load(os.path.join(base, 'models','crop_recommendation_model.pkl'))
label_encoder = joblib.load(os.path.join(base, 'models','label_encoder.pkl'))

model_fertilizer = joblib.load(os.path.join(base, 'models', 'fertilizer_recommendation_model.pkl'))
label_encoder_fertilizer = joblib.load(os.path.join(base, 'models', 'fertilizer_label_encoder.pkl'))
app.secret_key = 'your_secret_key'  # Set a secret key for session management

#Session
app.secret_key = 'Abhishek Chourasia 29122003' 

# Configure MySQL connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:1234@localhost/farmer_details'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Farmer model for MySQL
class Farmer(db.Model):
    __tablename__ = 'farmer'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(10), unique=True, nullable=False)
    username = db.Column(db.String(45), unique=True, nullable=False)
    password = db.Column(db.String(45), nullable=False)
    verified = db.Column(db.Boolean, default=False)  # To track if the farmer is verified

    def __init__(self, name, email, phone, username, password):
        self.name = name
        self.email = email
        self.phone = phone
        self.username = username
        self.password = password

# Admin model for MySQL
class Admin(db.Model):
    __tablename__ = 'admin'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(45), unique=True, nullable=False)
    password = db.Column(db.String(45), nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.password = password

class FarmerActivity(db.Model):
    __tablename__ = 'farmer_activity'
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey('farmer.id'), nullable=False)
    activity_type = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.Text, nullable=False)  # Store as JSON or plain text
    output_data = db.Column(db.Text, nullable=False) # Store as JSON or plain text
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)
    deleted = db.Column(db.Boolean, default=False)

    # Relationship with Farmer model
    farmer = db.relationship('Farmer', backref=db.backref('activities', lazy=True))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home_crop')
def home_crop():
    if 'farmer_id' not in session:
        flash("Please log in to access the home page.", "warning")
        return redirect(url_for('farmer_login'))
    
    farmer_id = session['farmer_id']
    farmer = Farmer.query.get(farmer_id)
    if not farmer:
        flash("Farmer not found. Please log in again.", "danger")
        return redirect(url_for('farmer_login'))
    
    return render_template('home_crop.html', farmer=farmer)


@app.route('/crop_detection')
def crop_detection():
    return render_template('crop_detection.html')

@app.route('/supplements', methods=['GET', 'POST'])
def supplements():
    return render_template('supplements.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        
        # Check file format based on extension
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return "Invalid file format. Please upload a valid image.", 400
        
        # Check file content type using imghdr to ensure it's a valid image
        image_type = imghdr.what(image)
        if image_type not in ['jpeg', 'png', 'jpg']:
            return "Invalid file format. Please upload a valid image.", 400

        filename = secure_filename(image.filename)  # Secure the filename
        file_path = os.path.join('static/uploads', filename)

        try:
            image.save(file_path)
            app.logger.debug(f"File saved at: {file_path}")
        except Exception as e:
            app.logger.error(f"Error saving file: {str(e)}")
            return f"Error saving file: {str(e)}", 500


        # Perform prediction
        pred = prediction(file_path)

        # Retrieve information about the predicted disease
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred] if 'image_url' in disease_info.columns else 'default_image_url'
        supplement_name = supplement_info['supplement name'][pred] if 'supplement name' in supplement_info.columns else 'N/A'
        supplement_image_url = supplement_info['supplement image'][pred] if 'supplement image' in supplement_info.columns else 'default_image_url'
        supplement_buy_link = supplement_info['buy link'][pred] if 'buy link' in supplement_info.columns else 'default_buy_link'

        formatted_string = "{\n"
        formatted_string += f"'Disease\n': '{title}',\n"
        formatted_string += f"'Description\n': '{description}',\n"
        formatted_string += f"'Prevent\n': '{prevent}'\n"
        formatted_string += "}"

        log_activity(session['farmer_id'], 'Crop Disease Detection', str({'image': filename}), formatted_string)

        # Pass details to the result page and include download URL
        return render_template('submit.html', title=title, desc=description, prevent=prevent, 
                               image_url=image_url, pred=pred, sname=supplement_name, 
                               simage=supplement_image_url, buy_link=supplement_buy_link,
                               download_url=f"/download_disease_report?title={title}&desc={description}&prevent={prevent}&sname={supplement_name}&simage={supplement_image_url}&buy_link={supplement_buy_link}&uploaded_image={file_path}")

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from io import BytesIO
import os
from flask import send_file, request

@app.route('/download_disease_report')
def download_disease_report():
    title = request.args.get('title')
    desc = request.args.get('desc')
    prevent = request.args.get('prevent')
    sname = request.args.get('sname')
    simage = request.args.get('simage')
    buy_link = request.args.get('buy_link')
    uploaded_image = request.args.get('uploaded_image')  # Path to the uploaded image

    # Path to the uploaded image for inclusion in the PDF
    uploaded_image_path = os.path.join(os.getcwd(), uploaded_image)
    supplement_image_path = os.path.join(os.getcwd(), simage)

    # Create a PDF in memory using SimpleDocTemplate for better layout management
    pdf_buffer = BytesIO()
    
    # Adjust the margin to reduce unnecessary space
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)

    # Create a list of elements to include in the PDF
    elements = []

    # Get sample styles for paragraphs
    styles = getSampleStyleSheet()

    # Create a custom style for justified text
    justified_style = ParagraphStyle(
        'JustifiedStyle',
        parent=styles['Normal'],
        alignment=4,  # 4 is for 'Justify'
        spaceAfter=12  # Space after each paragraph
    )

    # Custom style for green headings (bigger and bold)
    green_heading_style = ParagraphStyle(
        'GreenHeadingStyle',
        parent=styles['Heading1'],
        fontSize=16,
        fontName='Helvetica-Bold',
        textColor=colors.green,  # Green color
        alignment=1  # Center alignment for headings
    )

    green_heading_style_app = ParagraphStyle(
        'GreenHeadingStyle',
        parent=styles['Heading1'],
        fontSize=20,
        fontName='Helvetica-Bold',
        textColor=colors.green,  # Green color
        alignment=1  # Center alignment for headings
    )

    # Custom style for black paragraph text
    black_paragraph_style = ParagraphStyle(
        'BlackParagraphStyle',
        parent=justified_style,
        textColor=colors.black  # Black text color for paragraphs
    )

    title_paragraph_app = Paragraph(f"<b>KrishiDisha Disease Detection Report</b>", green_heading_style_app)
    elements.append(title_paragraph_app)
    elements.append(Spacer(1, 10))

    # Add title (Disease name) with the custom green style (centered)
    title_paragraph = Paragraph(f"<b>{title}</b>", green_heading_style)
    elements.append(title_paragraph)
    elements.append(Spacer(1, 12))  # Spacer between the title and the next content

    # Add uploaded image (crop image) between the title and description
    if os.path.exists(uploaded_image_path):
        try:
            crop_image = Image(uploaded_image_path, width=300, height=250)
            elements.append(crop_image)
            elements.append(Spacer(1, 6))  # Reduced spacer after crop image
        except Exception as e:
            print(f"Error adding crop image: {str(e)}")
            elements.append(Paragraph(f"<b>Error:</b> {str(e)}", styles['Normal']))
            elements.append(Spacer(1, 12))

    # Add description with black color paragraph and justified text
    desc_heading_paragraph = Paragraph("<b>Description:</b>", green_heading_style)
    elements.append(desc_heading_paragraph)
    elements.append(Spacer(1, 4))  # Reduced Spacer between heading and description text

    desc_paragraph = Paragraph(f"{desc}", black_paragraph_style)
    elements.append(desc_paragraph)
    elements.append(Spacer(1, 8))  # Reduced Spacer between description and the next content

    # Add preventive measures with black color paragraph and justified text
    prevent_heading_paragraph = Paragraph("<b>Preventive Measures:</b>", green_heading_style)
    elements.append(prevent_heading_paragraph)
    elements.append(Spacer(1, 4))  # Reduced Spacer between heading and preventive measures text

    prevent_paragraph = Paragraph(f"{prevent}", black_paragraph_style)
    elements.append(prevent_paragraph)
    elements.append(Spacer(1, 8))  # Reduced Spacer between preventive measures and next content

    # Add supplement name with black color paragraph and justified text
    supplement_name_heading_paragraph = Paragraph("<b>Supplement:</b>", green_heading_style)
    elements.append(supplement_name_heading_paragraph)
    elements.append(Spacer(1, 4))  # Reduced Spacer between heading and supplement name text

    supplement_name_paragraph = Paragraph(f"{sname}", black_paragraph_style)
    elements.append(supplement_name_paragraph)
    elements.append(Spacer(1, 8))  # Reduced Spacer between supplement name and next content

    # Add supplement image if exists
    if os.path.exists(supplement_image_path):
        supplement_image = Image(supplement_image_path, width=200, height=150)
        elements.append(supplement_image)
        elements.append(Spacer(1, 6))  # Reduced Spacer after supplement image

    # Add buy link with black color paragraph text
    buy_link_heading_paragraph = Paragraph("<b>Buy Link:</b>", green_heading_style)
    elements.append(buy_link_heading_paragraph)
    elements.append(Spacer(1, 4))  # Reduced Spacer between heading and buy link text

    buy_link_paragraph = Paragraph(f"<a href='{buy_link}'>{buy_link}</a>", black_paragraph_style)
    elements.append(buy_link_paragraph)
    elements.append(Spacer(1, 8))  # Reduced Spacer between buy link and next content

    # Build the PDF document
    doc.build(elements)

    # Move the buffer pointer to the beginning
    pdf_buffer.seek(0)

    # Send the PDF as a response
    return send_file(pdf_buffer, as_attachment=True, download_name="disease_report.pdf", mimetype="application/pdf")

try:
    data = pd.read_csv(data_path)
    unique_crops = sorted(data['Crop'].unique())
    unique_states = sorted(data['State'].unique())
    unique_seasons = sorted(data['Season'].unique())
except Exception as e:
    print(f"Error loading data: {e}")
    unique_crops, unique_states, unique_seasons = [], [], []

# Load the trained pipeline
try:
    pipeline = joblib.load(pipeline_path)
except Exception as e:
    print(f"Error loading pipeline: {e}")
    pipeline = None

# Crop yield prediction route

def log_activity(farmer_id, activity_type, input_data, output_data):
    try:
        activity = FarmerActivity(
            farmer_id=farmer_id,
            activity_type=activity_type,
            input_data=input_data,
            output_data=output_data
        )
        db.session.add(activity)
        db.session.commit()
    except Exception as e:
        app.logger.error(f"Error logging activity: {str(e)}")

import json
@app.route('/crop_yield', methods=['GET', 'POST']) 
def crop_yield():
    if request.method == 'POST':
        try:
            # Extract data from form submission
            crop = request.form['crop']
            crop_year = int(request.form['crop_year'])
            season = request.form['season']
            state = request.form['state']
            area = float(request.form['area'])
            production = float(request.form['production'])
            annual_rainfall = float(request.form['annual_rainfall'])
            fertilizer = float(request.form['fertilizer'])
            pesticide = float(request.form['pesticide'])

            # Prepare input data
            input_data = {
                'Crop': crop,
                'Crop_Year': crop_year,
                'Season': season,
                'State': state,
                'Area': area,
                'Production': production,
                'Annual_Rainfall': annual_rainfall,
                'Fertilizer': fertilizer,
                'Pesticide': pesticide
            }
            input_df = pd.DataFrame([input_data])

            # Predict yield
            if pipeline is None:
                raise ValueError("Prediction pipeline is not available.")
            prediction = pipeline.predict(input_df)
            predicted_yield = prediction[0]

            # Log activity
            farmer_id = session.get('farmer_id')  # Assuming the farmer ID is stored in the session
            if farmer_id:
                log_activity(
                    farmer_id=farmer_id,
                    activity_type="Crop Yield Prediction",
                    input_data=json.dumps(input_data),  # Convert dict to JSON string
                    output_data=json.dumps({'Predicted_Yield': predicted_yield})
                )

            return render_template(
                'crop_yield.html',
                prediction=f"{predicted_yield:.2f}",
                crops=unique_crops,
                states=unique_states,
                seasons=unique_seasons
            )
        except Exception as e:
            flash(f"An error occurred during prediction: {e}", "error")
            return render_template(
                'crop_yield.html',
                crops=unique_crops,
                states=unique_states,
                seasons=unique_seasons
            )

    # Render initial form for GET requests
    return render_template(
        'crop_yield.html',
        crops=unique_crops,
        states=unique_states,
        seasons=unique_seasons
    )



@app.route('/farmer_registration', methods=['GET', 'POST'])
def farmer_registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        username = request.form['username']
        password = request.form['password']

        new_farmer = Farmer(name=name, email=email, phone=phone, username=username, password=password)

        try:
            db.session.add(new_farmer)
            db.session.commit()
            flash('Registration successful! Please wait for admin verification.', 'success')
            return redirect(url_for('farmer_login'))
        except IntegrityError:
            db.session.rollback()
            flash('Username or email already exists. Please choose another.', 'danger')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')

    return render_template('farmer_registration.html')

# Route for Farmer Login
@app.route('/farmer_login', methods=['GET', 'POST'])
def farmer_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check farmer credentials
        farmer = Farmer.query.filter_by(username=username, password=password).first()
        if farmer:
            if farmer.verified:
                flash(f'{farmer.username} Login successful!', 'success')
                session['farmer_id'] = farmer.id  # Store farmer's ID in session
                return redirect(url_for('home_crop'))  # Redirect to home page after login
            else:
                flash(f'{farmer.username} your account is not verified by the admin. Please wait for verification.', 'warning')
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('farmer_login.html')
season_data = data.groupby('Season').size().reset_index(name='Count').to_dict('records')

# Calculate state_data
#Farmer Dashborads
@app.route('/farmer_dashboard', methods=['GET', 'POST'])
def farmer_dashboard():
    if 'farmer_id' not in session:
        flash("Please log in to access your dashboard.", "warning")
        return redirect(url_for('farmer_login'))

    farmer_id = session['farmer_id']
    farmer = Farmer.query.get(farmer_id)

    if not farmer:
        flash("Farmer not found. Please log in again.", "danger")
        return redirect(url_for('farmer_login'))

    if request.method == 'POST':
        activity_id = request.form.get('activity_id')
        activity = FarmerActivity.query.get(activity_id)
        if activity:
            activity.deleted = True
            db.session.commit()
            flash('Activity deleted successfully!', 'success')
        else:
            flash('Activity not found.', 'danger')

    activities = FarmerActivity.query.filter_by(farmer_id=farmer_id, deleted=False).order_by(FarmerActivity.timestamp.desc()).all()

    # File paths
    base = r'D:\KrishiDisha\Crop and Fertilizer Recommendation - Copy'
    crop_yield = os.path.join(base, 'data','crop_yield.csv')
    fertilizer_data = os.path.join(base, 'data', 'Fertilizer Prediction.csv')
    disease_data = os.path.join(base, 'data','disease_info.csv')
    supplement_file = os.path.join(base, 'data','supplement_info.csv')
    crop_recommendation_file = os.path.join(base, 'data','Crop_recommendation.csv')
    state_crop_file = os.path.join(base,'data','state_crop_production.csv')
    season_crop_file = os.path.join(base,'data','season_crop_stats.csv')

    try:
        crop_yield_df = pd.read_csv(crop_yield, encoding='ISO-8859-1')
        fertilizer_data_df = pd.read_csv(fertilizer_data, encoding='ISO-8859-1')
        disease_data_df = pd.read_csv(disease_data, encoding='ISO-8859-1')
        supplement_data_df = pd.read_csv(supplement_file, encoding='ISO-8859-1')
        crop_recommendation_df = pd.read_csv(crop_recommendation_file, encoding='ISO-8859-1')
        state_crop_df = pd.read_csv(state_crop_file,encoding='ISO-8859-1')
        season_crop_df = pd.read_csv(season_crop_file,encoding='ISO-8859-1')

        # Example data for charts (replace with actual computations)
        crop_yield_stats = crop_yield_df.groupby('Crop')['Yield'].mean().to_dict()
        fertilizer_stats = fertilizer_data_df['Fertilizer Name'].value_counts().to_dict()
        disease_stats = disease_data_df['disease_name'].value_counts().to_dict()

        # Overview Stats
        total_crops = crop_yield_df['Crop'].nunique()
        unique_diseases = disease_data_df['disease_name'].nunique()
        avg_yield = crop_yield_df['Yield'].mean()

        # Crop Yield Stats
        top_crops_by_yield = crop_yield_df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(5).to_dict()
        yield_distribution = crop_yield_df['Yield'].value_counts().to_dict()

        # Fertilizer Stats
        top_fertilizers = fertilizer_data_df['Fertilizer Name'].value_counts().to_dict()
        crop_fertilizer = fertilizer_data_df.groupby('Crop Type')['Fertilizer Name'].count().to_dict()

        # Disease Stats
        frequent_diseases = disease_data_df['disease_name'].value_counts().head(5).to_dict()
        severity_levels = disease_data_df['Possible Steps'].value_counts().to_dict()

        # Supplement Stats
        unique_supplements = supplement_data_df['supplement name'].nunique()
        supplements_by_disease = supplement_data_df.groupby('disease_name')['supplement name'].count().to_dict()
        missing_buy_links = supplement_data_df['buy link'].isna().sum()
        missing_images = supplement_data_df['supplement image'].isna().sum()

        # Crop Recommendation Stats
        recommended_crops = crop_recommendation_df['label'].value_counts().to_dict()
        crops_by_temperature = crop_recommendation_df.groupby(
            pd.cut(crop_recommendation_df['temperature'], bins=[0, 20, 30, 50])
        )['label'].count().to_dict()

        # State Crop Stats
        state_production_stats = state_crop_df.groupby('Top_State')['Total_Production'].sum().sort_values(ascending=False).to_dict()
        top_crops_by_state = state_crop_df.groupby('Top_State')['Crop'].apply(lambda x: x.value_counts().idxmax()).to_dict()

        # Season Crop Stats
        season_crop_stats = season_crop_df.groupby('Season')['Count'].sum().sort_values(ascending=False).to_dict()
        top_crops_by_season = season_crop_df.groupby('Season')['Crop'].apply(lambda x: x.value_counts().idxmax()).to_dict()

    except Exception as e:
        flash(f"Error loading data: {str(e)}", "danger")
        total_crops = unique_diseases = avg_yield = 0
        top_crops_by_yield = yield_distribution = top_fertilizers = crop_fertilizer = {}
        frequent_diseases = severity_levels = unique_supplements = {}
        supplements_by_disease = recommended_crops = crops_by_temperature = {}
        state_production_stats = top_crops_by_state = {}
        season_crop_stats = top_crops_by_season = {}
        missing_buy_links = missing_images = 0

    return render_template(
        'farmer_dashboard.html',
        farmer=farmer,
        activities=activities,
        crop_yield_stats=crop_yield_stats,
        fertilizer_stats=fertilizer_stats,
        disease_stats=disease_stats,
        total_crops=total_crops,
        unique_diseases=unique_diseases,
        avg_yield=avg_yield,
        topCropsByYield=top_crops_by_yield,
        yieldDistribution=yield_distribution,
        topFertilizers=top_fertilizers,
        cropFertilizer=crop_fertilizer,
        frequentDiseases=frequent_diseases,
        severityLevels=severity_levels,
        uniqueSupplements=unique_supplements,
        supplementsByDisease=supplements_by_disease,
        missingBuyLinks=missing_buy_links,
        missingImages=missing_images,
        recommendedCrops=recommended_crops,
        cropsByTemperature=crops_by_temperature,
        stateProductionStats=state_production_stats,
        topCropsByState=top_crops_by_state,
        seasonCropStats=season_crop_stats,
        topCropsBySeason=top_crops_by_season,
    )





# Continue adding functions for `create_farmer` after user-defined functionalities.

# Route to Create Farmer
@app.route('/create_farmer', methods=['GET', 'POST'])
def create_farmer():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        username = request.form['username']
        password = request.form['password']
        new_farmer = Farmer(name=name, email=email, phone=phone, username=username, password=password)
        

        try:
            db.session.add(new_farmer)
            db.session.commit()
            flash(f'Farmer {username} account created successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        except IntegrityError:
            db.session.rollback()
            flash('Username or Email already exists.', 'danger')
        except Exception as e:
            flash(f"Error occurred: {str(e)}", 'error')
    
    return render_template('create_farmer.html')

# Route for Admin Login
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        admin = Admin.query.filter_by(username=username, password=password).first()
        if admin:
            session['admin_logged_in'] = True
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials. Please try again.', 'danger')

    return render_template('admin_login.html')

@app.route('/create_admin', methods=['GET', 'POST'])
def create_admin():
    if 'admin_logged_in' not in session:
        flash('You must be logged in as an admin to create a new admin.', 'danger')
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        new_admin = Admin(username=username, password=password)

        try:
            db.session.add(new_admin)
            db.session.commit()
            flash(f'Admin account for "{username}" created successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        except IntegrityError:
            db.session.rollback()
            flash('Username already exists. Please choose a different one.', 'danger')
        except Exception as e:
            flash(f"Error occurred: {str(e)}", 'danger')

    return render_template('create_admin.html')


# Route for Admin Dashboard
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))

    farmers = Farmer.query.all()  # Get all farmers for verification
    return render_template('admin_dashboard.html', farmers=farmers)

# Route to verify farmer
@app.route('/verify_farmer/<int:farmer_id>')
def verify_farmer(farmer_id):
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))

    farmer = Farmer.query.get(farmer_id)
    if farmer:
        farmer.verified = True
        db.session.commit()
        flash(f'Farmer {farmer.username} has been verified.', 'success')
    else:
        flash(f'Farmer {farmer.username} not found.', 'danger')

    return redirect(url_for('admin_dashboard'))

# Route for Crop and Fertilizer Recommendation
@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    predicted_crop = None  # Initialize predicted_crop
    total_revenue = None  # Initialize total_revenue
    total_profit = None  # Initialize total_profit
    crop_image = None  # Initialize crop_image
    farmer_inputs = {}  # Store the farmer's input values
    
    if request.method == 'POST':
        try:
            # Extract input features from the form
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Store the farmer's input values in the session
            farmer_inputs = {
                'N': N,
                'P': P,
                'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
            session['farmer_inputs'] = farmer_inputs  # Save inputs to session

            # Create an input array
            input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            # Make a prediction
            prediction = model.predict(input_features)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]

            # Add your crop details (price per acre, cost per acre, image path)
            crop_details = {
                'rice': {'price_per_acre': 60000, 'cost_per_acre': 6000, 'image': 'images/rice.jpeg'},
        'maize': {'price_per_acre': 30000, 'cost_per_acre': 5000, 'image': 'images/maize.png'},
        'jute': {'price_per_acre': 45000, 'cost_per_acre': 7000, 'image': 'images/jute.jpg'},
        'cotton': {'price_per_acre': 65000, 'cost_per_acre': 8000, 'image': 'images/cotton.png'},
        'coconut': {'price_per_acre': 55000, 'cost_per_acre': 4000, 'image': 'images/coconut.jpg'},
        'papaya': {'price_per_acre': 150000, 'cost_per_acre': 30000, 'image': 'images/papaya.png'},
        'orange': {'price_per_acre': 120000, 'cost_per_acre': 25000, 'image': 'images/orange.png'},
        'apple': {'price_per_acre': 200000, 'cost_per_acre': 35000, 'image': 'images/apple.jpg'},
        'muskmelon': {'price_per_acre': 90000, 'cost_per_acre': 15000, 'image': 'images/muskmelon.png'},
        'watermelon': {'price_per_acre': 100000, 'cost_per_acre': 18000, 'image': 'images/watermelon.png'},
        'grapes': {'price_per_acre': 175000, 'cost_per_acre': 40000, 'image': 'images/grapes.png'},
        'mango': {'price_per_acre': 120000, 'cost_per_acre': 20000, 'image': 'images/mango.png'},
        'banana': {'price_per_acre': 140000, 'cost_per_acre': 25000, 'image': 'images/banana.png'},
        'pomegranate': {'price_per_acre': 160000, 'cost_per_acre': 30000, 'image': 'images/pomegranate.png'},
        'lentil': {'price_per_acre': 45000, 'cost_per_acre': 8000, 'image': 'images/lentil.png'},
        'blackgram': {'price_per_acre': 40000, 'cost_per_acre': 6000, 'image': 'images/blackgram.png'},
        'mungbean': {'price_per_acre': 50000, 'cost_per_acre': 7000, 'image': 'images/mungbean.png'},
        'mothbeans': {'price_per_acre': 35000, 'cost_per_acre': 5000, 'image': 'images/mothbeans.png'},
        'pigeonpeas': {'price_per_acre': 55000, 'cost_per_acre': 9000, 'image': 'images/pip.png'},
        'kidneybeans': {'price_per_acre': 60000, 'cost_per_acre': 10000, 'image': 'images/kindneybeans.png'},
        'chickpea': {'price_per_acre': 55000, 'cost_per_acre': 8500, 'image': 'images/chickpeas.jpg'},
        'coffee': {'price_per_acre': 200000, 'cost_per_acre': 40000, 'image': 'images/coffee.jpg'}
            }

            # Fetch details of the predicted crop
            if predicted_crop in crop_details:
                crop_info = crop_details[predicted_crop]
                price_per_acre = crop_info['price_per_acre']
                cost_per_acre = crop_info['cost_per_acre']
                crop_image = crop_info['image']
                
                # Calculate total revenue and total profit
                total_revenue = price_per_acre * 1  # You can change '1' to the acreage value entered by the user if needed
                total_profit = total_revenue - cost_per_acre

                farmer_id = session.get('farmer_id', 1)  # Assuming the farmer's ID is stored in session

                activity = FarmerActivity(
                farmer_id=farmer_id,
                activity_type='Crop Recommendation',
                input_data=str(farmer_inputs),
                output_data=f"Recommended Crop: {predicted_crop}, Revenue: {total_revenue}, Profit: {total_profit}"
                )
                db.session.add(activity)
                db.session.commit()

            else:
                return f"Crop details for {predicted_crop} not found.", 404

        except KeyError as e:
            return f"Missing data: {str(e)}", 400

        except ValueError as e:
            return f"Invalid input data: {str(e)}", 400

        except Exception as e:
            return f"An error occurred: {str(e)}", 500

    return render_template('crop_recommendation.html', prediction=predicted_crop, total_revenue=total_revenue, total_profit=total_profit, crop_image=crop_image, farmer_inputs=farmer_inputs)
"""
crop_details = {
        'rice': {'price_per_acre': 60000, 'cost_per_acre': 6000, 'image': 'static/images/rice.jpeg'},
        'maize': {'price_per_acre': 30000, 'cost_per_acre': 5000, 'image': 'static/images/maize.png'},
        'jute': {'price_per_acre': 45000, 'cost_per_acre': 7000, 'image': 'static/images/jute.jpg'},
        'cotton': {'price_per_acre': 65000, 'cost_per_acre': 8000, 'image': 'static/images/cotton.png'},
        'coconut': {'price_per_acre': 55000, 'cost_per_acre': 4000, 'image': 'static/images/coconut.jpg'},
        'papaya': {'price_per_acre': 150000, 'cost_per_acre': 30000, 'image': 'static/images/papaya.png'},
        'orange': {'price_per_acre': 120000, 'cost_per_acre': 25000, 'image': 'static/images/orange.png'},
        'apple': {'price_per_acre': 200000, 'cost_per_acre': 35000, 'image': 'static/images/apple.jpg'},
        'muskmelon': {'price_per_acre': 90000, 'cost_per_acre': 15000, 'image': 'static/images/muskmelon.png'},
        'watermelon': {'price_per_acre': 100000, 'cost_per_acre': 18000, 'image': 'static/images/watermelon.png'},
        'grapes': {'price_per_acre': 175000, 'cost_per_acre': 40000, 'image': 'static/images/grapes.png'},
        'mango': {'price_per_acre': 120000, 'cost_per_acre': 20000, 'image': 'static/images/mango.png'},
        'banana': {'price_per_acre': 140000, 'cost_per_acre': 25000, 'image': 'static/images/banana.png'},
        'pomegranate': {'price_per_acre': 160000, 'cost_per_acre': 30000, 'image': 'static/images/pomegranate.png'},
        'lentil': {'price_per_acre': 45000, 'cost_per_acre': 8000, 'image': 'static/images/lentil.png'},
        'blackgram': {'price_per_acre': 40000, 'cost_per_acre': 6000, 'image': 'static/images/blackgram.png'},
        'mungbean': {'price_per_acre': 50000, 'cost_per_acre': 7000, 'image': 'static/images/mungbean.png'},
        'mothbeans': {'price_per_acre': 35000, 'cost_per_acre': 5000, 'image': 'static/images/mothbeans.png'},
        'pigeonpeas': {'price_per_acre': 55000, 'cost_per_acre': 9000, 'image': 'static/images/pigeonpeas.png'},
        'kidneybeans': {'price_per_acre': 60000, 'cost_per_acre': 10000, 'image': 'static/images/kidneybeans.png'},
        'chickpea': {'price_per_acre': 55000, 'cost_per_acre': 8500, 'image': 'static/images/chickpeas.jpg'},
        'coffee': {'price_per_acre': 200000, 'cost_per_acre': 40000, 'image': 'static/images/coffee.jpg'}
    }"""
import os
from reportlab.platypus import Spacer
from reportlab.platypus import Image

@app.route('/download_report', methods=['POST'])
def download_report():
    # Get the predicted crop and farmer inputs from the session
    predicted_crop = request.form.get('predicted_crop')
    farmer_inputs = session.get('farmer_inputs')
    

    if not farmer_inputs:
        return "Farmer inputs not found!", 400
    

        

    # Crop details with price per acre, cost per acre, and image path

    STATIC_DIR = r"D:\KrishiDisha\Crop and Fertilizer Recommendation - Copy\static\images"

# Update crop_details with absolute paths
    crop_details = {
    'rice': {'price_per_acre': 60000, 'cost_per_acre': 6000, 'image': os.path.join(STATIC_DIR, 'rice.jpeg')},
    'maize': {'price_per_acre': 30000, 'cost_per_acre': 5000, 'image': os.path.join(STATIC_DIR, 'maize.png')},
    'jute': {'price_per_acre': 45000, 'cost_per_acre': 7000, 'image': os.path.join(STATIC_DIR, 'jute.jpg')},
    'cotton': {'price_per_acre': 65000, 'cost_per_acre': 8000, 'image': os.path.join(STATIC_DIR, 'cotton.png')},
    'coconut': {'price_per_acre': 55000, 'cost_per_acre': 4000, 'image': os.path.join(STATIC_DIR, 'coconut.jpg')},
    'papaya': {'price_per_acre': 150000, 'cost_per_acre': 30000, 'image': os.path.join(STATIC_DIR, 'papaya.png')},
    'orange': {'price_per_acre': 120000, 'cost_per_acre': 25000, 'image': os.path.join(STATIC_DIR, 'orange.png')},
    'apple': {'price_per_acre': 200000, 'cost_per_acre': 35000, 'image': os.path.join(STATIC_DIR, 'apple.jpg')},
    'muskmelon': {'price_per_acre': 90000, 'cost_per_acre': 15000, 'image': os.path.join(STATIC_DIR, 'muskmelon.png')},
    'watermelon': {'price_per_acre': 100000, 'cost_per_acre': 18000, 'image': os.path.join(STATIC_DIR, 'watermelon.png')},
    'grapes': {'price_per_acre': 175000, 'cost_per_acre': 40000, 'image': os.path.join(STATIC_DIR, 'grapes.png')},
    'mango': {'price_per_acre': 120000, 'cost_per_acre': 20000, 'image': os.path.join(STATIC_DIR, 'mango.png')},
    'banana': {'price_per_acre': 140000, 'cost_per_acre': 25000, 'image': os.path.join(STATIC_DIR, 'banana.png')},
    'pomegranate': {'price_per_acre': 160000, 'cost_per_acre': 30000, 'image': os.path.join(STATIC_DIR, 'pomegranate.png')},
    'lentil': {'price_per_acre': 45000, 'cost_per_acre': 8000, 'image': os.path.join(STATIC_DIR, 'lentil.png')},
    'blackgram': {'price_per_acre': 40000, 'cost_per_acre': 6000, 'image': os.path.join(STATIC_DIR, 'blackgram.png')},
    'mungbean': {'price_per_acre': 50000, 'cost_per_acre': 7000, 'image': os.path.join(STATIC_DIR, 'mungbean.png')},
    'mothbeans': {'price_per_acre': 35000, 'cost_per_acre': 5000, 'image': os.path.join(STATIC_DIR, 'mothbeans.png')},
    'pigeonpeas': {'price_per_acre': 55000, 'cost_per_acre': 9000, 'image': os.path.join(STATIC_DIR, 'pip.png')},
    'kidneybeans': {'price_per_acre': 60000, 'cost_per_acre': 10000, 'image': os.path.join(STATIC_DIR, 'kindneybeans.png')},
    'chickpea': {'price_per_acre': 55000, 'cost_per_acre': 8500, 'image': os.path.join(STATIC_DIR, 'chickpeas.jpg')},
    'coffee': {'price_per_acre': 200000, 'cost_per_acre': 40000, 'image': os.path.join(STATIC_DIR, 'coffee.jpg')},
}
    

    # Get the details of the predicted crop
    crop_info = crop_details.get(predicted_crop)
    if not crop_info:
        return "Crop details not found!", 404

    price_per_acre = crop_info['price_per_acre']
    cost_per_acre = crop_info['cost_per_acre']
    image_path = crop_info['image']

    # Create a PDF report
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Add title with styling
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    title_style.fontSize = 18
    title_style.fontName = "Helvetica-Bold"
    title_style.alignment = 1  # Center the title
    title_paragraph = Paragraph(f"<font size=18>KrishiDisha Crop Recommendation Report</font>", title_style)
    elements.append(title_paragraph)

    # Add user input data in a table format
    user_data = [['Field', 'Farmer Input Value']]
    for key, value in farmer_inputs.items():
        user_data.append([key.capitalize(), str(value)])

    user_table = Table(user_data, colWidths=[120, 200])
    user_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(user_table)
    elements.append(Spacer(1, 25))
    # Add crop recommendation details in another table format
    crop_data = [['Field', 'Details']]
    crop_data.append(['Predicted Crop', predicted_crop.capitalize()])
    crop_data.append(['Price per Acre', f"Rs{price_per_acre}"])
    crop_data.append(['Cost per Acre', f"Rs{cost_per_acre}"])
    crop_data.append(['Total Revenue', f"Rs{price_per_acre}"])
    crop_data.append(['Total Profit', f"Rs{price_per_acre - cost_per_acre}"])

    crop_table = Table(crop_data, colWidths=[120, 200])
    crop_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    elements.append(crop_table)
    elements.append(Spacer(1, 20))
    # Add the crop image at the end with a sufficient size
    # Add the crop image at the end with a sufficient size
    # Add the crop image at the end with a sufficient size
    if os.path.exists(image_path):
        crop_image = Image(image_path, width=400, height=300)  # reportlab.platypus.Image
        crop_image.hAlign = 'CENTER'
        elements.append(crop_image)

    else:
        app.logger.error(f"Image not found: {image_path}")
        elements.append(Paragraph("<font color='red'>Image not found!</font>", styles['Normal']))


    # Build the PDF
    doc.build(elements)

    # Save the PDF to the buffer
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="crop_recommendation_report.pdf", mimetype="application/pdf")

@app.route('/fertilizer_recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    # Define mappings at the start of the function
    soil_mapping = {
        0: 'Black',
        1: 'Clayey',
        2: 'Loamy',
        3: 'Red',
        4: 'Sandy'
    }
    
    crop_mapping = {
        0: 'Barley',
        1: 'Cotton',
        2: 'Ground Nuts',
        3: 'Maize',
        4: 'Millets',
        5: 'Oil seeds',
        6: 'Paddy',
        7: 'Pulses',
        8: 'Sugarcane',
        9: 'Tobacco',
        10: 'Wheat'
    }

    # Fertilizer mapping dictionary with image paths
    fertilizer_mapping = {
        0: '10-26-26',
        1: '14-35-14',
        2: '17-17-17',
        3: '20-20',
        4: '28-28',
        5: 'DAP',
        6: 'Urea'
    }
    
    fertilizer_images = {
        0: 'static/images/fertilizers/10-26-26.png',
        1: 'static/images/fertilizers/14-35-14.png',
        2: 'static/images/fertilizers/17-17-17.png',
        3: 'static/images/fertilizers/20-20.png',
        4: 'static/images/fertilizers/28-28.png',
        5: 'static/images/fertilizers/dap.jpg',
        6: 'static/images/fertilizers/Urea.jpg'
    }

    if request.method == 'POST':
        try:
            # Get form data
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            moisture = float(request.form['moisture'])
            soil_type = int(request.form['soil_type'])
            crop_type = int(request.form['crop_type'])
            N = float(request.form['N'])
            K = float(request.form['K'])
            P = float(request.form['P'])

            # Prepare input data
            input_data = np.array([[temperature, humidity, moisture, soil_type, crop_type, N, K, P]])
            
            # Make prediction
            prediction_fertilizer = model_fertilizer.predict(input_data)[0]  # Get the prediction
            
            # Validate prediction
            predicted_fertilizer = fertilizer_mapping.get(int(prediction_fertilizer), "Unknown Fertilizer")
            predicted_fertilizer_image = fertilizer_images.get(int(prediction_fertilizer), "static/images/crop1.png")
            session['farmer_inputs'] = {
    'temperature': temperature,
    'humidity': humidity,
    'moisture': moisture,
    'soil_type': soil_mapping.get(soil_type),
    'crop_type': crop_mapping.get(crop_type),
    'N': N,
    'K': K,
    'P': P
}

            log_activity(session['farmer_id'], 'Fertilizer Recommendation', 
                         str({
                             'temperature': temperature, 
                             'humidity': humidity,
                             'moisture': moisture,
                             'soil_type': soil_mapping.get(soil_type),
                             'crop_type': crop_mapping.get(crop_type),
                             'N': N, 'K': K, 'P': P
                         }),
                         str({'Fertilizer Recommendation': predicted_fertilizer}))

            # Render result
            return render_template('fertilizer_recommendation.html', 
                                   prediction_fertilizer=predicted_fertilizer,
                                   fertilizer_image=predicted_fertilizer_image,
                                   crop_mapping=crop_mapping,
                                   soil_mapping=soil_mapping)

        except Exception as e:
            print(f"Error: {e}")
            return render_template('fertilizer_recommendation.html', 
                                   prediction_fertilizer="An error occurred during prediction.",
                                   crop_mapping=crop_mapping,
                                   soil_mapping=soil_mapping)
    
    # If the request method is GET, render the form with the mappings
    return render_template('fertilizer_recommendation.html', 
                           crop_mapping=crop_mapping,
                           soil_mapping=soil_mapping)

@app.route('/download_fertilizer_report', methods=['POST'])
def download_fertilizer_report():
    # Define STATIC_DIR for absolute image paths
    STATIC_DIR = r"D:\KrishiDisha\Crop and Fertilizer Recommendation - Copy\static\images\fertilizers"

    # Update fertilizer_details with absolute paths for the images
    fertilizer_details = {
        '10-26-26': {'image': os.path.join(STATIC_DIR, '10-26-26.png')},
        '14-35-14': {'image': os.path.join(STATIC_DIR, '14-35-14.png')},
        '17-17-17': {'image': os.path.join(STATIC_DIR, '17-17-17.png')},
        '20-20': {'image': os.path.join(STATIC_DIR, '20-20.png')},
        '28-28': {'image': os.path.join(STATIC_DIR, '28-28.png')},
        'DAP': {'image': os.path.join(STATIC_DIR, 'DAP.jpg')},
        'Urea': {'image': os.path.join(STATIC_DIR, 'Urea.jpg')}
    }

    # Retrieve inputs from session and form
    farmer_inputs = session.get('farmer_inputs')
    predicted_fertilizer = request.form.get('predicted_fertilizer')

    # Validate required data
    if not farmer_inputs or not predicted_fertilizer:
        app.logger.error("Missing farmer inputs or predicted fertilizer!")
        return "Missing required information!", 400

    # Get the details of the predicted fertilizer
    fertilizer_info = fertilizer_details.get(predicted_fertilizer)
    if not fertilizer_info:
        return "Fertilizer details not found!", 404

    fertilizer_image_path = fertilizer_info['image']

    # Create a PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Add Title
    styles = getSampleStyleSheet()
    title = Paragraph(f"<font size=18>KrishiDisha Fertilizer Recommendation Report</font>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 20))

    # Add Farmer Inputs
    elements.append(Paragraph("<b>Farmer Inputs:</b>", styles['Heading2']))
    input_data = [['Field', 'Value']]
    for key, value in farmer_inputs.items():
        input_data.append([key.capitalize(), str(value)])
    input_table = Table(input_data, colWidths=[150, 350])
    input_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(input_table)
    elements.append(Spacer(1, 10))

    # Add Fertilizer Recommendation
    elements.append(Paragraph("<b>Fertilizer Recommendation:</b>", styles['Heading2']))
    fertilizer_data = [
        ['Recommended Fertilizer', predicted_fertilizer],
    ]
    fertilizer_table = Table(fertilizer_data, colWidths=[200, 300])
    fertilizer_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(fertilizer_table)

    # Add Fertilizer Image
    if os.path.exists(fertilizer_image_path):
        elements.append(Spacer(1, 20))
        img = Image(fertilizer_image_path, width=400, height=300)
        img.hAlign = 'CENTER'
        elements.append(img)

    # Generate and return PDF
    doc.build(elements)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="fertilizer_recommendation_report.pdf",
        mimetype="application/pdf"
    )





'''@app.route('/create_farmer', methods=['GET', 'POST'])
def create_farmer():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        username = request.form['username']
        password = request.form['password']  # You might want to hash this for security

        new_farmer = Farmer(name=name, email=email, phone=phone, username=username, password=password)
        db.session.add(new_farmer)
        db.session.commit()
        flash('Farmer created successfully!', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('create_farmer.html')'''

@app.route('/update_farmer/<int:farmer_id>', methods=['GET', 'POST'])
def update_farmer(farmer_id):
    farmer = Farmer.query.get_or_404(farmer_id)
    if request.method == 'POST':
        farmer.name = request.form['name']
        farmer.email = request.form['email']
        farmer.phone = request.form['phone']
        farmer.username = request.form['username']
        db.session.commit()
        flash('Farmer information updated successfully!', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('update_farmer.html', farmer=farmer)

from sqlalchemy.exc import IntegrityError

@app.route('/delete_farmer/<int:farmer_id>')
def delete_farmer(farmer_id):
    try:
        farmer = Farmer.query.get_or_404(farmer_id)

        # Delete related farmer activities
        FarmerActivity.query.filter_by(farmer_id=farmer_id).delete()

        # Now delete the farmer
        db.session.delete(farmer)
        db.session.commit()

        flash('Farmer deleted successfully!', 'success')

    except IntegrityError:
        db.session.rollback()
        flash('Error: Could not delete farmer due to existing related data.', 'danger')

    return redirect(url_for('admin_dashboard'))


# Route to logout admin
@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('admin_login'))

@app.route('/farmer_logout')
def farmer_logout():
    # Remove farmer session data
    session.pop('farmer_id', None)  # Remove farmer_id from session
    flash('Logged out successfully.', 'success')
    return redirect(url_for('farmer_login'))

# Add to imports
import random

# Add these to your existing imports
from flask import jsonify, request
import re  # Add if not already present

# Add after other routes
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.form['message']
        farmer_id = session.get('farmer_id')  # Get farmer ID from session
        response = generate_chat_response(user_message, farmer_id)
        return jsonify({'response': response})
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({'response': "I'm currently experiencing technical difficulties. Please try again later."}), 500

def generate_chat_response(message, farmer_id=None):
    message = message.lower().strip()

    # Pre-login responses
    if not farmer_id:
        return handle_pre_login_queries(message)

    # Post-login responses
    return handle_post_login_queries(message, farmer_id)

def handle_pre_login_queries(message):
    responses = {
        "hello":"Hello, I am KrishiDisha Bot how can i help you!",
        "what is krishidisha": "KrishiDisha is an AI-powered agricultural platform that provides crop disease detection, yield prediction, fertilizer recommendations, and farming analytics to empower farmers.",
        "how to use": "1. Register/Login 2. Upload crop images for disease detection 3. Use crop/fertilizer recommender 4. Check yield predictions 5. Access your dashboard",
        "admin approval time": "Admin verification typically takes 24-48 hours. Your account will be activated once verified.",
        "features": "Key features: 🌱 Real-time disease detection 🌾 Crop yield predictions 💊 Fertilizer recommendations 📊 Farming analytics",
        "contact": "Reach us at support@krishidisha.com or call +91-XXXXXX7890",
        "login help": "Visit the login page and use your registered credentials. Reset password via admin if needed.",
    }

    for key in responses:
        if key in message:
            return responses[key]
    
    return "I'm here to help with agricultural advice! How can I assist you today?"

def handle_post_login_queries(message, farmer_id):
    farmer = Farmer.query.get(farmer_id)
    if not farmer:
        return "Farmer not found. Please log in again."

    # Greet the user by name
    if any(greeting in message for greeting in ["hi", "hello", "hey"]):
        return f"Hello {farmer.name}! How can I assist you today?"

    # Crop Recommendation Patterns
    if re.search(r'\b(crop|recommend|soil|weather|plant|sow|harvest)\b', message):
        return generate_crop_recommendation_response(message, farmer_id)

    # Disease Detection Patterns
    if re.search(r'\b(detect|disease|leaf|plant sick|symptom|yellow|brown|spots)\b', message):
        return generate_disease_detection_response(message)

    # Fertilizer Patterns
    if re.search(r'\b(fertilizer|nutrient|npk|soil fertility|soil health)\b', message):
        return generate_fertilizer_recommendation_response(message)

    # General Farming Advice
    if re.search(r'\b(how to grow|cultivation|farming tips|planting|irrigation)\b', message):
        crop = extract_crop_name(message)
        if crop:
            return get_crop_advice(crop)
        return "I can provide cultivation advice! Please mention a crop name (e.g., 'How to grow rice?')."

    # Weather/Climate Patterns
    if re.search(r'\b(weather|rainfall|climate|temperature|humidity)\b', message):
        return generate_weather_response(message)

    # Default Responses
    return random.choice([
        "I'm here to help with agricultural advice! How can I assist you today?",
        "Feel free to ask about crops, diseases, fertilizers, or farming practices!",
        "Let's grow together! Ask me about crop management or agricultural technologies."
    ])

def generate_crop_recommendation_response(message, farmer_id):
    if re.search(r'\b(crop recommendation|N|P|K|temperature|humidity|ph|rainfall)\b', message):
        return ("Based on your soil parameters, I recommend visiting our crop recommendation page "
                "for accurate predictions. You can also share your values here for quick advice.")
    
    return ("I can help with crop recommendations! Please visit our crop recommendation page or "
            "tell me your soil parameters (N, P, K values, temperature, humidity, pH, rainfall).")

def generate_disease_detection_response(message):
    if re.search(r'\b(detect|disease|upload|image|photo|picture)\b', message):
        return ("For disease detection, please upload an image of the affected plant on our disease "
                "detection page. I can help analyze it!")
    
    return ("If your plants are showing signs of disease, such as yellowing leaves or spots, "
            "please upload an image for analysis. You can also visit our supplements page for remedies.")

def generate_fertilizer_recommendation_response(message):
    if re.search(r'\b(N|P|K|soil type|crop type)\b', message):
        return ("Based on your soil and crop details, I recommend visiting our fertilizer recommendation page "
                "for accurate suggestions. You can also share your values here for quick advice.")
    
    return ("I can recommend fertilizers based on your soil conditions. Please visit our fertilizer "
            "recommendation page or share your soil details (N, P, K values, soil type, crop type).")

def generate_weather_response(message):
    if re.search(r'\b(location|region|state)\b', message):
        return ("To provide accurate weather advice, please share your location or state. "
                "Our system considers local weather patterns for crop recommendations.")
    
    return ("Optimal crop growth depends on weather conditions. Our system considers annual rainfall "
            "and temperature in predictions. Share your location for more specific advice.")

def extract_crop_name(message):
    crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'coffee', 'mango', 'banana', 'apple', 'grapes']
    for crop in crops:
        if crop in message:
            return crop
    return None

def get_crop_advice(crop):
    advice = {
        'rice': "Rice requires flooded fields. Maintain 5-10cm water depth and pH 5-6.5. Use nitrogen-rich fertilizers.",
        'wheat': "Wheat grows best in well-drained loamy soil with pH 6-7.5. Rotate crops to maintain soil health.",
        'maize': "Maize needs warm temperatures (18-27°C) and well-drained soil. Use balanced NPK fertilizers.",
        'cotton': "Cotton requires long frost-free periods. Maintain soil pH 6-7. Use phosphorus-heavy fertilizers.",
        'sugarcane': "Sugarcane needs tropical climate with 1500-2500mm annual rainfall. Regular irrigation is crucial.",
        'coffee': "Coffee plants need shade and well-drained volcanic soil. Maintain pH 6-6.5 and moderate temperatures.",
        'mango': "Mango trees thrive in well-drained soil with pH 5.5-7.5. Provide regular irrigation and organic mulch.",
        'banana': "Banana plants require rich, well-drained soil with pH 6-7.5. Maintain high humidity and regular watering.",
        'apple': "Apple trees need well-drained loamy soil with pH 6-7. Prune regularly and ensure proper sunlight.",
        'grapes': "Grapes grow best in well-drained sandy loam soil with pH 6-7. Provide trellis support and regular pruning."
    }
    return advice.get(crop.lower(), "Please specify a valid crop for detailed advice.")


# After app creation

if __name__ == '__main__':

    app.run(debug=True)
    