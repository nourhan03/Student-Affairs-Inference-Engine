from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()

# Student model
class Student(db.Model):
    __tablename__ = 'Students'
    Id = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(100), nullable=False)
    NationalId = db.Column(db.String(14), unique=True, nullable=False)
    Gender = db.Column(db.String(10), nullable=False)
    DateOfBirth = db.Column(db.Date, nullable=False)
    Address = db.Column(db.String(100))
    Nationality = db.Column(db.String(50))
    Email = db.Column(db.String(100), unique=True, nullable=False)
    Phone = db.Column(db.String(15), unique=True, nullable=False)
    Semester = db.Column(db.Integer, nullable=False)
    EnrollmentDate = db.Column(db.Date, nullable=False)
    High_School_degree = db.Column(db.Numeric(10, 2), nullable=False)
    High_School_Section = db.Column(db.String(50), nullable=False)
    CreditsCompleted = db.Column(db.Integer, nullable=False)
    ImagePath = db.Column(db.String(255), nullable=True)
    DepartmentId = db.Column(db.Integer, db.ForeignKey('Departments.Id'), nullable=False)
    StudentLevel = db.Column(db.Integer)
    status = db.Column(db.String(50), nullable=False)
    GPA1 = db.Column( db.Float)
    GPA2 = db.Column( db.Float)
    GPA3 = db.Column( db.Float)
    GPA4 = db.Column( db.Float)
    GPA5  = db.Column( db.Float)
    GPA6 = db.Column( db.Float)
    GPA7 = db.Column( db.Float)
    GPA8 = db.Column( db.Float)
      

# Professor model
class Professor(db.Model):
    __tablename__ = 'Professors'
    Id = db.Column(db.Integer, primary_key=True)
    FullName = db.Column(db.String(100), nullable=False)
    NationalId = db.Column(db.String(14), unique=True, nullable=False)
    Gender = db.Column(db.String(10), nullable=False)
    DateOfBirth = db.Column(db.Date, nullable=False)
    Address = db.Column(db.String(100))
    Email = db.Column(db.String(100), unique=True, nullable=False)
    Phone = db.Column(db.String(15), unique=True, nullable=False)
    Join_Date = db.Column(db.Date, nullable=False)
    Position = db.Column(db.String(20), nullable=False)
    ImagePath = db.Column(db.String(255), nullable=True)
    DepartmentId = db.Column(db.Integer, db.ForeignKey('Departments.Id'), nullable=False)


# Course model
class Course(db.Model):
    __tablename__ = 'Courses'
    Id = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(50), nullable=False)
    Code = db.Column(db.String(50), nullable=False)
    Description = db.Column(db.String(250), nullable=False)
    Credits = db.Column(db.Integer, nullable=False)
    Status = db.Column(db.String(50), nullable=False)
    Semester = db.Column(db.Integer, nullable=False)
    PreCourseId = db.Column(db.Integer, db.ForeignKey('Courses.Id'))
    MaxSeats = db.Column(db.Integer, nullable=False)
    CurrentEnrolledStudents = db.Column(db.Integer, default=0)
    course_departments = db.relationship('CourseDepartment', backref='course')

# Class model
class Class(db.Model):
    __tablename__ = 'Classes'
    Id = db.Column(db.Integer, primary_key=True)
    StartTime = db.Column(db.Time, nullable=False)
    EndTime = db.Column(db.Time, nullable=False)
    Day = db.Column(db.String(20), nullable=False)
    Location = db.Column(db.String(100))
    ProfessorId = db.Column(db.Integer, db.ForeignKey('Professors.Id'), nullable=False)
    CourseId = db.Column(db.Integer, db.ForeignKey('Courses.Id'), nullable=False)


# Attendance model
class Attendance(db.Model):
    __tablename__ = 'Attendances'
    Id = db.Column(db.Integer, primary_key=True)
    Date = db.Column(db.DateTime, nullable=False)
    Status = db.Column(db.Boolean, nullable=False)
    ClassesId = db.Column(db.Integer, db.ForeignKey('Classes.Id'), nullable=False)
    StudentId = db.Column(db.Integer, db.ForeignKey('Students.Id'), nullable=False)

# Department model
class Department(db.Model):
    __tablename__ = 'Departments'
    Id = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(100), nullable=False)
    ProfessorCount = db.Column(db.Integer)
    HeadOfDepartment = db.Column(db.String(100))

# CourseDepartment model
class CourseDepartment(db.Model):
    __tablename__ = 'CourseDepartments'
    Id = db.Column(db.Integer, primary_key=True)
    CourseId = db.Column(db.Integer, db.ForeignKey('Courses.Id'), nullable=False)
    DepartmentId = db.Column(db.Integer, db.ForeignKey('Departments.Id'), nullable=False)
    IsMandatory = db.Column(db.Boolean, nullable=False)

# Enrollment model
class Enrollment(db.Model):
    __tablename__ = 'Enrollments'
    Id = db.Column(db.Integer, primary_key=True)
    Semester = db.Column(db.String, nullable=False)
    Exam1Grade = db.Column(db.Float)
    Exam2Grade = db.Column(db.Float)
    Grade = db.Column(db.Float)
    NumberOFSemster = db.Column(db.String, nullable=False)
    StudentId = db.Column(db.Integer, db.ForeignKey('Students.Id'), nullable=False)
    CourseId = db.Column(db.Integer, db.ForeignKey('Courses.Id'), nullable=False)
    AddedEnrollmentDate = db.Column(db.Date)
    DeletedEnrollmentDate = db.Column(db.Date)
    IsCompleted = db.Column(db.String(50))
