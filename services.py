from models import db, Student, Course, Attendance, Class, Department, Enrollment, CourseDepartment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import joinedload
from datetime import datetime
from redis_config import redis_client



from functools import lru_cache
from models import db, Student, Course, Attendance, Class, Department, Enrollment, CourseDepartment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import joinedload
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    pass

@lru_cache(maxsize=128)
def get_student_data(student_id):
    """
    الحصول على بيانات الطالب مع التخزين المؤقت
    
    Args:
        student_id (int): معرف الطالب

    Returns:
        dict: بيانات الطالب

    Raises:
        ValidationError: في حالة عدم وجود الطالب أو البيانات غير صحيحة
    """
    try:
        student = Student.query.filter_by(Id=student_id).first()
        if not student:
            raise ValidationError(f"Student with ID {student_id} not found")

        current_gpa = getattr(student, f'GPA{student.Semester}', 0.0)

        completed_courses = db.session.query(Enrollment.CourseId).filter(
            Enrollment.StudentId == student.Id,
            Enrollment.IsCompleted == 'ناجح'
        ).all()
        
        failed_courses = db.session.query(Enrollment.CourseId).filter(
            Enrollment.StudentId == student.Id,
            Enrollment.IsCompleted == 'راسب'
        ).all()

        return {
            "id": student.Id,
            "name": student.Name,
            "department_id": student.DepartmentId,
            "current_semester": student.Semester,
            "gpa": current_gpa,
            "completed_courses": [c[0] for c in completed_courses],
            "failed_courses": [c[0] for c in failed_courses]
        }
    except Exception as e:
        logger.error(f"Error in get_student_data: {str(e)}")
        raise

@lru_cache(maxsize=128)
def get_prerequisites():
    """
    الحصول على المتطلبات السابقة للمواد مع التخزين المؤقت
    
    Returns:
        dict: قاموس يحتوي على المتطلبات السابقة لكل مادة
    """
    try:
        prerequisites = {}
        courses = db.session.query(Course.Id, Course.PreCourseId).all()

        for course_id, pre_course_id in courses:
            prerequisites[course_id] = [pre_course_id] if pre_course_id else []

        return prerequisites
    except Exception as e:
        logger.error(f"Error in get_prerequisites: {str(e)}")
        raise

@lru_cache(maxsize=128)
def get_course_data():
    """
    الحصول على بيانات المواد مع التخزين المؤقت
    
    Returns:
        dict: قاموس يحتوي على بيانات كل مادة
    """
    try:
        courses = (db.session.query(Course)
                  .options(joinedload(Course.course_departments))
                  .all())
        
        result = {}
        for course in courses:
            course_departments = course.course_departments

            if course_departments:
                for cd in course_departments:
                    result[course.Id] = {
                        "id": course.Id,
                        "name": course.Name,
                        "code": course.Code,
                        "description": course.Description,
                        "semester": course.Semester,
                        "department_id": cd.DepartmentId,
                        "is_mandatory": bool(cd.IsMandatory)
                    }
            else:
                result[course.Id] = {
                    "id": course.Id,
                    "name": course.Name,
                    "code": course.Code,
                    "description": course.Description,
                    "semester": course.Semester,
                    "department_id": None,
                    "is_mandatory": False
                }
        
        return result
    except Exception as e:
        logger.error(f"Error in get_course_data: {str(e)}")
        raise

def get_available_courses(semester, department_id):
    """
    الحصول على المواد المتاحة للفصل الدراسي والقسم
    
    Args:
        semester (int): الفصل الدراسي
        department_id (int): معرف القسم

    Returns:
        list: قائمة بمعرفات المواد المتاحة
    """
    try:
        if not semester or not department_id:
            raise ValidationError("Semester and department_id are required")

        courses = (db.session.query(Course)
                 .join(CourseDepartment, Course.Id == CourseDepartment.CourseId)
                 .filter(Course.Semester == semester, 
                        CourseDepartment.DepartmentId == department_id, 
                        Course.Status == 'نشط')
                 .all())
        return [course.Id for course in courses]
    except Exception as e:
        logger.error(f"Error in get_available_courses: {str(e)}")
        raise

def get_registerable_courses(student_data, available_courses, course_data):
    """
    الحصول على المواد التي يمكن للطالب تسجيلها
    
    Args:
        student_data (dict): بيانات الطالب
        available_courses (list): المواد المتاحة
        course_data (dict): بيانات المواد

    Returns:
        list: قائمة بالمواد التي يمكن تسجيلها
    """
    try:
        if not all([student_data, available_courses, course_data]):
            raise ValidationError("Missing required data for course registration")

        current_semester = student_data.get("current_semester")
        student_department = student_data.get("department_id")

        if not current_semester or not student_department:
            raise ValidationError("Missing semester or department information")

        current_semester_courses = [
            course for course in available_courses
            if course_data.get(course, {}).get("semester") == current_semester
            and student_department == course_data.get(course, {}).get("department_id")
        ]

        failed_mandatory_courses = [
            course for course in student_data["failed_courses"]
            if course_data.get(course, {}).get("is_mandatory") == True
            and student_department == course_data.get(course, {}).get("department_id")
        ]

        registerable_courses = list(set(current_semester_courses + failed_mandatory_courses))
        return registerable_courses

    except Exception as e:
        logger.error(f"Error in get_registerable_courses: {str(e)}")
        raise

def recommend_courses(student_data, available_courses, course_data, prerequisites):
    """
    توصية المواد للطالب
    
    Args:
        student_data (dict): بيانات الطالب
        available_courses (list): المواد المتاحة
        course_data (dict): بيانات المواد
        prerequisites (dict): المتطلبات السابقة

    Returns:
        dict: قاموس يحتوي على المواد الموصى بها (إجبارية واختيارية)
    """
    try:
        if not all([student_data, available_courses, course_data, prerequisites]):
            raise ValidationError("Missing required data for course recommendation")

        registerable_courses = get_registerable_courses(student_data, available_courses, course_data)

        course_descriptions = {
            course_id: course_data.get(course_id, {}).get("description", "")
            for course_id in course_data
        }

        eligible_courses = [
            course for course in registerable_courses
            if all(prereq in student_data["completed_courses"] 
                  for prereq in prerequisites.get(course, []))
        ]

        mandatory_courses = [
            course for course in eligible_courses
            if course_data.get(course, {}).get("is_mandatory") == True
        ]
        
        elective_courses = [
            course for course in eligible_courses
            if course_data.get(course, {}).get("is_mandatory") == False
        ]

        if not elective_courses:
            return {
                "mandatory": mandatory_courses,
                "elective": []
            }

        studied_courses = student_data["completed_courses"]
        studied_descriptions = [course_descriptions.get(course, "") for course in studied_courses]
        available_descriptions = [course_descriptions.get(course, "") for course in elective_courses]

        if not any(studied_descriptions) or not any(available_descriptions):
            return {
                "mandatory": mandatory_courses,
                "elective": elective_courses
            }

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(studied_descriptions + available_descriptions)
        similarity_matrix = cosine_similarity(
            tfidf_matrix[:len(studied_descriptions)], 
            tfidf_matrix[len(studied_descriptions):]
        )
        similarity_scores = similarity_matrix.mean(axis=0)

        course_similarity = list(zip(elective_courses, similarity_scores))
        sorted_courses = [course for course, _ in sorted(
            course_similarity, 
            key=lambda x: x[1], 
            reverse=True
        )]

        return {
            "mandatory": mandatory_courses,
            "elective": sorted_courses
        }

    except Exception as e:
        logger.error(f"Error in recommend_courses: {str(e)}")
        raise

def get_current_semester():
    """الحصول على الفصل الدراسي الحالي بناءً على التاريخ الحالي"""
    current_date = datetime.now()
    current_month = current_date.month
    current_year = current_date.year
    
    # طباعة معلومات تصحيح
    print(f"Current month: {current_month}, Current year: {current_year}")
    
    # تحديد الفصل الدراسي بناءً على الشهر
    # الفصل الخريفي (Fall): من سبتمبر إلى يناير (9-1)
    # الفصل الربيعي (Spring): من فبراير إلى يونيو (2-6)
    # الفصل الصيفي (Summer): يوليو وأغسطس (7-8)
    
    semester_name = ""
    semester_number = 0
    
    if 2 <= current_month <= 6:
        # الفصل الربيعي (Spring)
        semester_name = "Spring"
        semester_number = 2
    elif 9 <= current_month <= 12:
        # الفصل الخريفي (Fall)
        semester_name = "Fall"
        semester_number = 1
    elif current_month == 1:
        # الفصل الخريفي (Fall) من العام السابق
        semester_name = "Fall"
        semester_number = 1
        # نستخدم العام السابق لشهر يناير لأنه جزء من الفصل الخريفي للعام السابق
        current_year -= 1
    else:
        # الفصل الصيفي (Summer)
        semester_name = "Summer"
        semester_number = 3
    
    # إضافة السنة إلى اسم الفصل
    full_semester_name = f"{semester_name} {current_year}"
    
    print(f"Determined semester: {full_semester_name} ({semester_number})")
    
    return semester_number, full_semester_name

def get_recommended_courses(student_id):
    """الحصول على المواد الموصى بها للطالب"""
    try:
        # 1. التحقق من وجود الطالب
        student = db.session.query(Student).get(student_id)
        if not student:
            return None

        # 2. الحصول على المواد المتاحة للطالب في الترم الحالي
        current_semester, _ = get_current_semester()
        available_courses = db.session.query(Course)\
            .join(CourseDepartment, Course.Id == CourseDepartment.CourseId)\
            .filter(
                CourseDepartment.DepartmentId == student.DepartmentId,
                Course.Status == 'نشط',
                Course.Semester == current_semester
            ).all()
            
        # 3. الحصول على بيانات الطالب
        student_data = get_student_data(student_id)
        
        # 4. الحصول على المتطلبات السابقة وبيانات المواد
        prerequisites = get_prerequisites()
        course_data = get_course_data()
        
        # 5. الحصول على المواد المتاحة كقائمة معرفات
        available_course_ids = [course.Id for course in available_courses]
        
        # 6. الحصول على التوصيات
        recommendations = recommend_courses(
            student_data,
            available_course_ids,
            course_data,
            prerequisites
        )
        
        # 7. دمج المواد الإلزامية والاختيارية
        all_recommended_courses = []
        
        # إضافة المواد الإلزامية
        for course_id in recommendations["mandatory"]:
            course = course_data.get(course_id, {})
            all_recommended_courses.append({
                "id": course_id,
                "name": course.get("name", "Unknown"),
                "code": course.get("code", "Unknown"),
                "description": course.get("description", ""),
                "is_mandatory": True
            })
        
        # إضافة المواد الاختيارية
        for course_id in recommendations["elective"]:
            course = course_data.get(course_id, {})
            all_recommended_courses.append({
                "id": course_id,
                "name": course.get("name", "Unknown"),
                "code": course.get("code", "Unknown"),
                "description": course.get("description", ""),
                "is_mandatory": False
            })
        
        return all_recommended_courses
        
    except Exception as e:
        logger.error(f"Error in get_recommended_courses: {str(e)}")
        return None

def check_enrollment_period():
    """التحقق من فترة التسجيل"""
    start_time = redis_client.get('enrollment:start_time')
    end_time = redis_client.get('enrollment:end_time')
    
    if not start_time or not end_time:
        return False, "لم يتم تعيين فترة التسجيل"
        
    now = datetime.now()
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    
    if now < start_dt:
        return False, "لم يبدأ التسجيل بعد"
    elif now > end_dt:
        return False, "انتهت فترة التسجيل"
        
    return True, None

