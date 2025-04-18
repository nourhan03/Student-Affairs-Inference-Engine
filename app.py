from flask import Flask
from flask_restful import Api
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from models import db
from redis_config import redis_client
from datetime import datetime
import logging
import os
from flask_cors import CORS
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    CORS(app)
    api = Api(app)


    
    app.config['SECRET_KEY'] = 'your-secret-key-here' 
    
    
    app.config['SQLALCHEMY_DATABASE_URI'] = (
        "mssql+pyodbc:///"
        "FacultyManagementDB"
        "?driver=ODBC+Driver+17+for+SQL+Server"
        "&trusted_connection=yes"
        "&server=localhost"
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    try:
        db.init_app(app)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")

    
    from resources import (
        RecommendCourses, CourseEnrollment, DeleteEnrollment, 
        EnrollmentPeriod, EnrollmentPeriodStatus,
        GraduationEligibility, GraduationRequirements,
        AcademicPerformanceEvaluation,RecommendCoursesWithCredits,
        GraduationCheckResource
    )

    # endpoints
    api.add_resource(RecommendCourses, '/recommend-courses/<int:student_id>')
    

    api.add_resource(EnrollmentPeriod, '/enrollment-period')
    api.add_resource(EnrollmentPeriodStatus, '/enrollment-period/status')

    api.add_resource(GraduationEligibility, '/graduation-check/<int:student_id>')
    api.add_resource(GraduationRequirements, '/graduation-requirements/<int:student_id>')

    api.add_resource(CourseEnrollment, '/enrollment/add/<int:student_id>')
    api.add_resource(DeleteEnrollment, '/enrollment/delete/<int:student_id>')

    api.add_resource(AcademicPerformanceEvaluation, '/academic-evaluation/<int:student_id>')

    api.add_resource(GraduationCheckResource, '/graduation-check/<int:student_id>')

    return app

if __name__ == '__main__':
    app = create_app()
    try:
        logger.info("Starting the server...")
        app.run(debug=False, port=5000)
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")






