from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from functools import lru_cache
import json
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

from datetime import datetime
from flask_restful import Resource, request
from flask import jsonify
from sqlalchemy.orm import joinedload
from sqlalchemy import func

from redis_config import redis_client
from services import (
    get_student_data, get_prerequisites, get_course_data,
    get_available_courses, get_registerable_courses, get_recommended_courses,
    recommend_courses, ValidationError, check_enrollment_period, get_current_semester
)
from models import db, Student, Course, Attendance, Class, Department, Enrollment, CourseDepartment, Professor

import logging
import pickle

logger = logging.getLogger(__name__)


class RecommendCourses(Resource):
    @lru_cache(maxsize=32)
    def get(self, student_id):
        try:
            
            student_data = get_student_data(student_id)
            if not student_data:
                return {"error": "Student not found"}, 404

            logger.debug(f"Student data: {student_data}")

           
            available_courses = get_available_courses(
                student_data["current_semester"],
                student_data["department_id"]
            )
            
            logger.debug(f"Available courses: {available_courses}")

            
            prerequisites = get_prerequisites()
            course_data = get_course_data()

            logger.debug(f"Prerequisites: {prerequisites}")
            logger.debug(f"Course data: {course_data}")

            
            if not available_courses:
                return {
                    "recommendations": []
                }

            result = recommend_courses(
                student_data,
                available_courses,
                course_data,
                prerequisites
            )

            
            all_courses = []
            
            
            for course_id in result["mandatory"]:
                course_info = self._format_course(course_id, course_data)
                course_info["نوع_المادة"] = "اجباري"
                
                
                course_info["سبب_الاقتراح"] = self._get_mandatory_reason(course_id, student_data, prerequisites)
                
                # إضافة عدد الساعات
                course_obj = Course.query.get(course_id)
                if course_obj and hasattr(course_obj, 'Credits'):
                    course_info["credits"] = course_obj.Credits
                
                all_courses.append(course_info)
            
            
            for course_id in result["elective"]:
                course_info = self._format_course(course_id, course_data)
                course_info["نوع_المادة"] = "اختياري"
                
                
                completed_courses = student_data.get("completed_courses", [])
                similarity_score = self._calculate_similarity(course_id, completed_courses, course_data)
                course_info["درجة_التشابه"] = similarity_score
                course_info["سبب_الاقتراح"] = self._get_elective_reason(course_id, student_data, similarity_score)
                
                
                course_obj = Course.query.get(course_id)
                if course_obj and hasattr(course_obj, 'Credits'):
                    course_info["credits"] = course_obj.Credits
                
                all_courses.append(course_info)

            response = {
                "recommendations": all_courses
            }

            return jsonify(response)

        except ValidationError as e:
            logger.warning(f"Validation error for student {student_id}: {str(e)}")
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Error processing recommendation for student {student_id}: {str(e)}")
            return {"error": str(e)}, 500

    @staticmethod
    def _format_course(course_id, course_data):
        """تنسيق بيانات المادة مع معلومات المحاضرات والمقاعد المتاحة"""
        course = course_data.get(course_id, {})
        
        logger.debug(f"Fetching class info for course ID: {course_id}")
        
        course_details = {
            "id": course_id,
            "name": course.get("name", "غير محدد"),
            "code": course.get("code", "غير محدد"),
            "description": course.get("description", "غير محدد"),
        }
        
        # الحصول على معلومات المقاعد
        course_obj = Course.query.get(course_id)
        if course_obj:
            max_seats = course_obj.MaxSeats
            
            current_enrolled = db.session.query(db.func.count(Enrollment.Id)).filter(
                Enrollment.CourseId == course_id,
                Enrollment.IsCompleted == "قيد الدراسة",
                Enrollment.DeletedEnrollmentDate == None
            ).scalar()
            
            available_seats = max_seats - current_enrolled
            course_details["المقاعد_المتاحة"] = available_seats
        
        # التحقق من وجود محاضرات لهذه المادة
        class_count = db.session.query(db.func.count(Class.Id)).filter(Class.CourseId == course_id).scalar()
        logger.debug(f"Class count for course {course_id}: {class_count}")
        
        # الحصول على معلومات المحاضرة
        class_info = Class.query.filter_by(CourseId=course_id).first()
        
        if class_info:
            # طباعة معلومات تصحيح
            logger.debug(f"Found class info for course {course_id}: {class_info.Id}, Day: {class_info.Day}, ProfessorId: {class_info.ProfessorId}")
            
            # الحصول على معلومات الأستاذ بشكل منفصل
            professor = Professor.query.get(class_info.ProfessorId)
            if professor:
                logger.debug(f"Found professor: {professor.FullName}")
                professor_name = professor.FullName
            else:
                logger.debug(f"No professor found for ID: {class_info.ProfessorId}")
                professor_name = "غير محدد"
            
            # تحديد المكان بناءً على البيانات المتاحة
            location = class_info.Location if hasattr(class_info, 'Location') and class_info.Location else "غير محدد"
            
            course_details["معلومات_المحاضرة"] = {
                "اليوم": class_info.Day,
                "وقت_البداية": str(class_info.StartTime),
                "وقت_النهاية": str(class_info.EndTime),
                "المكان": location,
                "الدكتور": professor_name
            }
        else:
            # طباعة معلومات تصحيح
            logger.debug(f"No class info found for course {course_id}")
            
            # إذا لم يتم العثور على معلومات المحاضرة، نضع قيم فارغة
            course_details["معلومات_المحاضرة"] = {
                "اليوم": "غير محدد",
                "وقت_البداية": "غير محدد",
                "وقت_النهاية": "غير محدد",
                "المكان": "غير محدد",
                "الدكتور": "غير محدد"
            }

        return course_details
        
    @staticmethod
    def _calculate_similarity(course_id, completed_courses, course_data):
        """حساب درجة التشابه بين المادة والمواد المكتملة"""
        if not completed_courses:
            return 0.0
            
        course_desc = course_data.get(course_id, {}).get("description", "")
        completed_descs = [course_data.get(c_id, {}).get("description", "") for c_id in completed_courses]
        
        if not course_desc or not any(completed_descs):
            return 0.0
            
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([course_desc] + completed_descs)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            return float(np.mean(similarity))
        except:
            return 0.0
    
    @staticmethod
    def _get_mandatory_reason(course_id, student_data, prerequisites):
        """تحديد سبب اقتراح المادة الإجبارية"""
        course_prereqs = prerequisites.get(course_id, [])
        
        is_graduation_req = course_id in student_data.get("graduation_requirements", [])
        
        is_prereq_for_others = any(course_id in prereqs for c_id, prereqs in prerequisites.items())
        
        reasons = []
        
        if is_graduation_req:
            reasons.append("مادة إجبارية للتخرج")
        
        if is_prereq_for_others:
            reasons.append("متطلب سابق لمواد أخرى")
        
        if course_prereqs:
            completed_prereqs = [p for p in course_prereqs if p in student_data.get("completed_courses", [])]
            if len(completed_prereqs) == len(course_prereqs):
                reasons.append("تم استيفاء جميع المتطلبات السابقة")
        
        _, current_semester_name = get_current_semester()
        reasons.append(f"مناسبة للفصل الدراسي الحالي ({current_semester_name})")
        
        if not reasons:
            return "مادة إجبارية في الخطة الدراسية"
        
        return " - ".join(reasons)
    
    @staticmethod
    def _get_elective_reason(course_id, student_data, similarity_score):
        """تحديد سبب اقتراح المادة الاختيارية"""
        reasons = []
        
        if similarity_score > 0.7:
            reasons.append("تشابه كبير مع المواد السابقة")
        elif similarity_score > 0.4:
            reasons.append("تشابه متوسط مع المواد السابقة")
        elif similarity_score > 0.1:
            reasons.append("تشابه قليل مع المواد السابقة")
        
        gpa = student_data.get("gpa", 0)
        if gpa and gpa > 3.5:
            reasons.append("مناسبة للطلاب ذوي المعدل المرتفع")
        elif gpa and gpa > 2.5:
            reasons.append("مناسبة للطلاب ذوي المعدل المتوسط")
        
        semester = student_data.get("current_semester", 0)
        if semester > 6:
            reasons.append("مناسبة للطلاب في المراحل المتقدمة")
        elif semester > 3:
            reasons.append("مناسبة للطلاب في المراحل المتوسطة")
        
        if not reasons:
            return "مادة اختيارية متاحة للتسجيل"
        
        return " - ".join(reasons)

class EnrollmentPeriod(Resource):
    def post(self):
        """تعيين فترة التسجيل"""
        try:
            # التحقق من وجود بيانات JSON في الطلب
            if not request.is_json:
                return {"error": "يجب إرسال البيانات بتنسيق JSON"}, 400
                
            data = request.json
            start_time = data.get('start_time')  # "2024-03-10 08:00:00"
            end_time = data.get('end_time')      # "2024-03-20 23:59:59"
            
            if not start_time or not end_time:
                return {"error": "يجب تحديد وقت البداية والنهاية"}, 400
            
            # التحقق من صحة التواريخ
            try:
                start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return {"error": "صيغة التاريخ غير صحيحة. يجب أن تكون بالصيغة: YYYY-MM-DD HH:MM:SS"}, 400
            
            if end_dt <= start_dt:
                return {"error": "يجب أن يكون وقت النهاية بعد وقت البداية"}, 400
            
            # حفظ في Redis
            try:
                redis_client.set('enrollment:start_time', start_time)
                redis_client.set('enrollment:end_time', end_time)
            except Exception as e:
                logger.error(f"Redis error: {str(e)}")
                return {"error": "حدث خطأ أثناء حفظ فترة التسجيل"}, 500
            
            return {
                "message": "تم تعيين فترة التسجيل بنجاح",
                "period": {
                    "start_time": start_time,
                    "end_time": end_time
                }
            }, 201
        except Exception as e:
            logger.error(f"Error setting enrollment period: {str(e)}")
            return {"error": f"حدث خطأ: {str(e)}"}, 500
    
    def get(self):
        """الحصول على فترة التسجيل الحالية"""
        try:
            start_time = redis_client.get('enrollment:start_time')
            end_time = redis_client.get('enrollment:end_time')
            
            if not start_time or not end_time:
                return {"error": "لم يتم تعيين فترة التسجيل"}, 404
                
            return {
                "period": {
                    "start_time": start_time,
                    "end_time": end_time
                }
            }, 200
        except Exception as e:
            logger.error(f"Error getting enrollment period: {str(e)}")
            return {"error": f"حدث خطأ: {str(e)}"}, 500

class EnrollmentPeriodStatus(Resource):
    def get(self):
        """التحقق من حالة فترة التسجيل"""
        start_time = redis_client.get('enrollment:start_time')
        end_time = redis_client.get('enrollment:end_time')
        
        if not start_time or not end_time:
            return {
                "status": "غير متاح",
                "message": "لم يتم تعيين فترة التسجيل"
            }, 200
            
        now = datetime.now()
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        
        if now < start_dt:
            return {
                "status": "لم يبدأ",
                "message": "لم يبدأ التسجيل بعد",
                "starts_in": str(start_dt - now)
            }, 200
        elif now > end_dt:
            return {
                "status": "منتهي",
                "message": "انتهت فترة التسجيل",
                "ended_since": str(now - end_dt)
            }, 200
        else:
            return {
                "status": "متاح",
                "message": "التسجيل متاح حالياً",
                "remaining_time": str(end_dt - now)
            }, 200
        



class CourseEnrollment(Resource):
    def post(self, student_id):
        try:
            # التحقق من فترة التسجيل
            enrollment_active, message = check_enrollment_period()
            if not enrollment_active:
                return {"error": "لا يمكن إضافة مواد جديدة. " + message}, 400

            # الحصول على بيانات الطلب
            data = request.get_json()
            
            # طباعة معلومات تصحيح
            logger.debug(f"Received enrollment request: {data}")
            
            if not data or 'courses' not in data:
                return {"error": "لم يتم تحديد المواد المطلوبة"}, 400
                
            courses = data['courses']
            
            if not courses or not isinstance(courses, list):
                return {"error": "يجب تحديد قائمة المواد المطلوبة"}, 400

            # التحقق من وجود الطالب
            student = Student.query.get(student_id)
            if not student:
                return {"error": "الطالب غير موجود"}, 404
            
            # الحصول على المواد الموصى بها للطالب
            recommend_service = RecommendCourses()
            recommendations_response = recommend_service.get(student_id)
            
            # طباعة معلومات تصحيح
            logger.debug(f"Recommendations response type: {type(recommendations_response)}")
            logger.debug(f"Recommendations response: {recommendations_response}")
            
            # استخراج معرفات المواد الموصى بها
            recommended_course_ids = []
            
            # إذا كانت الاستجابة هي Response object (من jsonify)
            if hasattr(recommendations_response, 'json'):
                recommendations_data = recommendations_response.json
                if isinstance(recommendations_data, dict) and 'recommendations' in recommendations_data:
                    for course in recommendations_data['recommendations']:
                        if isinstance(course, dict) and 'id' in course:
                            recommended_course_ids.append(course['id'])
            # إذا كانت الاستجابة هي tuple (status code, response)
            elif isinstance(recommendations_response, tuple) and len(recommendations_response) == 2:
                recommendations_data = recommendations_response[0]
                if isinstance(recommendations_data, dict) and 'recommendations' in recommendations_data:
                    for course in recommendations_data['recommendations']:
                        if isinstance(course, dict) and 'id' in course:
                            recommended_course_ids.append(course['id'])
            
            # طباعة معلومات تصحيح
            logger.debug(f"Recommended course IDs: {recommended_course_ids}")
            logger.debug(f"Requested course IDs: {courses}")
            
            # التحقق من أن جميع المواد المطلوبة موجودة في قائمة التوصيات
            invalid_courses = [course_id for course_id in courses if course_id not in recommended_course_ids]
            if invalid_courses:
                return {
                    "error": "لا يمكن تسجيل بعض المواد لأنها غير موجودة في قائمة التوصيات",
                    "invalid_courses": invalid_courses
                }, 400
            
            # الحصول على الفصل الدراسي الحالي
            current_semester_number, semester_name = get_current_semester()
            
            # حساب عدد الساعات المسجلة حاليًا
            current_credits = get_current_enrolled_credits(student_id, semester_name)
            
            # حساب عدد الساعات المطلوبة
            requested_credits = 0
            for course_id in courses:
                course = Course.query.get(course_id)
                if course and hasattr(course, 'Credits'):
                    requested_credits += course.Credits
            
            # الحصول على الحد الأقصى للساعات المسموح بها
            max_credits = get_max_credits(student)
            
            # التحقق من عدم تجاوز الحد الأقصى للساعات
            if current_credits + requested_credits > max_credits:
                return {
                    "error": f"لا يمكن تسجيل هذه المواد. الحد الأقصى المسموح به هو {max_credits} ساعة.",
                    "current_credits": current_credits,
                    "requested_credits": requested_credits,
                    "max_credits": max_credits
                }, 400
            
            # تسجيل المواد
            enrollments = []
            for course_id in courses:
                # التحقق من عدم وجود تسجيل سابق للمادة
                existing_enrollment = Enrollment.query.filter_by(
                    StudentId=student_id,
                    CourseId=course_id,
                    Semester=semester_name,
                    DeletedEnrollmentDate=None  # فقط التسجيلات النشطة
                ).first()
                
                if existing_enrollment:
                    continue  # تخطي المواد المسجلة بالفعل
                
                # التحقق من وجود المادة
                course = Course.query.get(course_id)
                if not course:
                    continue  # تخطي المواد غير الموجودة
                
                # إنشاء تسجيل جديد
                enrollment = Enrollment(
                    StudentId=student_id,
                    CourseId=course_id,
                    Semester=semester_name,
                    NumberOFSemster=str(current_semester_number),
                    AddedEnrollmentDate=datetime.now().date(),
                    IsCompleted="قيد الدراسة"
                )
                db.session.add(enrollment)
                
                # تحديث عدد الطلاب المسجلين في المادة
                if hasattr(course, 'CurrentEnrolledStudents'):
                    course.CurrentEnrolledStudents += 1
                
                enrollments.append(course_id)
            
            if not enrollments:
                return {"message": "لم يتم تسجيل أي مواد جديدة. قد تكون المواد مسجلة بالفعل."}, 200
                
            # حفظ التغييرات
            db.session.commit()
            
            return {"message": f"تم تسجيل {len(enrollments)} مواد بنجاح", "enrolled_courses": enrollments}, 201
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error in course enrollment: {str(e)}")
            return {"error": str(e)}, 500

class DeleteEnrollment(Resource):
    def delete(self, student_id):
        try:
            # التحقق من فترة التسجيل
            enrollment_active, message = check_enrollment_period()
            if not enrollment_active:
                return {"error": "لا يمكن حذف المواد. " + message}, 400

            # الحصول على بيانات الطلب
            data = request.get_json()
            
            # طباعة معلومات تصحيح
            logger.debug(f"Received delete enrollment request: {data}")
            
            if not data:
                return {"error": "لم يتم تحديد المواد المطلوب حذفها"}, 400
            
            # التحقق من وجود المواد في البيانات (يقبل 'courses' أو 'course_id')
            courses = None
            if 'courses' in data:
                courses = data['courses']
            elif 'course_id' in data:
                courses = data['course_id']
            
            if not courses:
                return {"error": "لم يتم تحديد المواد المطلوب حذفها"}, 400
            
            # التحقق من نوع المواد
            if not isinstance(courses, list):
                # إذا كانت المواد ليست قائمة، نحولها إلى قائمة
                courses = [courses]
            
            # طباعة معلومات تصحيح
            logger.debug(f"Courses to delete: {courses}")

            # التحقق من وجود الطالب
            student = Student.query.get(student_id)
            if not student:
                return {"error": "الطالب غير موجود"}, 404
            
            # الحصول على الفصل الدراسي الحالي
            _, semester_name = get_current_semester()
            
            # حذف التسجيلات
            deleted_courses = []
            for course_id in courses:
                # البحث عن التسجيل
                enrollment = Enrollment.query.filter_by(
                    StudentId=student_id,
                    CourseId=course_id,
                    Semester=semester_name,
                    DeletedEnrollmentDate=None,  # فقط التسجيلات النشطة
                    IsCompleted="قيد الدراسة"    # فقط المواد قيد الدراسة
                ).first()
                
                if not enrollment:
                    continue  # تخطي المواد غير المسجلة
                
                # تحديث تاريخ الحذف وحالة المادة
                enrollment.DeletedEnrollmentDate = datetime.now().date()
                enrollment.IsCompleted = "تم الحذف"  # تحديث حالة المادة
                
                # تحديث عدد الطلاب المسجلين في المادة
                course = Course.query.get(course_id)
                if course and hasattr(course, 'CurrentEnrolledStudents') and course.CurrentEnrolledStudents > 0:
                    course.CurrentEnrolledStudents -= 1
                
                deleted_courses.append(course_id)
            
            if not deleted_courses:
                return {"message": "لم يتم حذف أي مواد. قد تكون المواد غير مسجلة أو تم حذفها بالفعل."}, 200
                
            # حفظ التغييرات
            db.session.commit()
            
            return {"message": f"تم حذف {len(deleted_courses)} مواد بنجاح", "deleted_courses": deleted_courses}, 200
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error in delete enrollment: {str(e)}")
            return {"error": str(e)}, 500

class GraduationEligibility(Resource):
    def get(self, student_id):
        try:
            # 1. التحقق من وجود الطالب
            student = db.session.query(Student).get(student_id)
            if not student:
                return {"error": "الطالب غير موجود"}, 404

            # 2. حساب مجموع الساعات المكتملة
            total_credits = db.session.query(func.sum(Student.CreditsCompleted))\
                .filter(Student.Id == student_id)\
                .scalar() or 0
            
            # 3. التحقق من المواد الإلزامية
            completed_courses = db.session.query(Enrollment, Course)\
                .join(Course, Enrollment.CourseId == Course.Id)\
                .filter(
                    Enrollment.StudentId == student_id,
                    Enrollment.IsCompleted == 'ناجح'
                ).all()

            mandatory_courses = db.session.query(Course)\
                .join(CourseDepartment, Course.Id == CourseDepartment.CourseId)\
                .filter(
                    CourseDepartment.DepartmentId == student.DepartmentId,
                    CourseDepartment.IsMandatory == True
                ).all()

            mandatory_course_ids = {course.Id for course in mandatory_courses}
            completed_course_ids = {course.Id for course, _ in completed_courses}
            
            missing_mandatory = mandatory_course_ids - completed_course_ids

            # 4. التحقق من الأهلية
            is_eligible = total_credits >= 136 and not missing_mandatory

            return {
                "student_id": student_id,
                "student_name": student.Name,
                "total_credits": total_credits,
                "required_credits": 136,
                "completed_mandatory_courses": len(mandatory_course_ids - missing_mandatory),
                "total_mandatory_courses": len(mandatory_course_ids),
                "is_eligible": is_eligible,
                "reasons": [
                    *(["الساعات المكتملة أقل من 136"] if total_credits < 136 else []),
                    *(["توجد مواد إلزامية غير مكتملة"] if missing_mandatory else [])
                ]
            }, 200

        except Exception as e:
            return {"error": f"حدث خطأ: {str(e)}"}, 500            



class GraduationRequirements(Resource):
    def get(self, student_id):
        try:
            # التحقق من وجود الطالب
            student = Student.query.get(student_id)
            if not student:
                return {"error": "الطالب غير موجود"}, 404
            
            # الحصول على معلومات القسم
            department = Department.query.get(student.DepartmentId)
            if not department:
                return {"error": "القسم غير موجود"}, 404
            
            # الحصول على الفصل الدراسي الحالي للطالب
            current_semester = student.Semester
            
            # الحصول على المعدل التراكمي الحالي للطالب
            current_gpa = self._get_current_gpa(student)
            
            # الحصول على المواد التي اجتازها الطالب
            completed_courses = self._get_completed_courses(student_id)
            
            # حساب عدد الساعات المكتملة
            if hasattr(student, 'CreditsCompleted') and student.CreditsCompleted is not None and student.CreditsCompleted > 0:
                completed_credits = student.CreditsCompleted
            else:
                # حساب الساعات المكتملة من المواد المكتملة
                completed_credits = sum(course.get('credits', 0) for course in completed_courses)
            
            # تحديد إجمالي الساعات المطلوبة للتخرج (ثابت 136 ساعة للبكالوريوس)
            total_required_credits = 136
            
            # حساب الساعات المتبقية
            remaining_credits = max(0, total_required_credits - completed_credits)
            
            # حساب نسبة الإنجاز
            completion_percentage = (completed_credits / total_required_credits) * 100 if total_required_credits > 0 else 0
            
            # تحديد حالة المعدل التراكمي
            gpa_status, gpa_message = self._check_gpa_status(current_gpa)
            
            # الحصول على المواد المتبقية المطلوبة
            remaining_courses = self._get_remaining_required_courses(student_id, department.Id)
            
            # تصنيف المواد المتبقية حسب النوع (إجباري، اختياري)
            categorized_remaining_courses = self._categorize_remaining_courses(remaining_courses)
            
            # تحديد المواد الموصى بها للفصل الدراسي القادم
            recommended_courses = self._get_recommended_next_courses(
                student_id, 
                remaining_courses, 
                completed_courses, 
                current_semester + 1  # الفصل القادم
            )
            
            # إنشاء توصيات للطالب
            recommendations = self._create_recommendations(
                student, 
                current_gpa, 
                completion_percentage, 
                remaining_credits
            )
            
            return {
                "student_info": {
                    "id": student.Id,
                    "name": student.Name,
                    "department": department.Name,
                    "current_semester": current_semester,
                    "current_gpa": float(current_gpa)  # تحويل المعدل إلى عدد عشري
                },
                "graduation_status": {
                    "completed_credits": completed_credits,
                    "remaining_credits": remaining_credits,
                    "total_required_credits": total_required_credits,
                    "completion_percentage": round(completion_percentage, 2)
                },
                "courses": {
                    "completed": completed_courses,
                    "remaining": categorized_remaining_courses,
                    "recommended_next": recommended_courses
                },
                "recommendations": recommendations
            }, 200
            
        except Exception as e:
            logger.error(f"Error in graduation requirements: {str(e)}")
            return {"error": f"حدث خطأ: {str(e)}"}, 500
    
    def _get_current_gpa(self, student):
        """
        الحصول على المعدل التراكمي الحالي للطالب
        """
        try:
            # الحصول على المعدل التراكمي
            if hasattr(student, 'GPA') and student.GPA is not None:
                return float(student.GPA)
            
            # إذا لم يكن هناك معدل في حقل GPA، نحاول حسابه من المواد المكتملة
            enrollments = Enrollment.query.filter_by(
                StudentId=student.Id,
                IsCompleted="ناجح"
            ).all()
            
            if not enrollments:
                return 0.0
            
            total_points = 0
            total_credits = 0
            
            for enrollment in enrollments:
                course = Course.query.get(enrollment.CourseId)
                if course and hasattr(course, 'Credits') and hasattr(enrollment, 'Grade'):
                    grade = float(enrollment.Grade) if enrollment.Grade is not None else 0
                    credits = float(course.Credits) if course.Credits is not None else 0
                    
                    # حساب النقاط
                    points = grade * credits
                    
                    total_points += points
                    total_credits += credits
            
            # حساب المعدل التراكمي
            if total_credits > 0:
                gpa = total_points / total_credits / 25  # تحويل من 100 إلى 4.0
                return round(gpa, 2)
            
            return 0.0
        except Exception as e:
            logger.error(f"Error getting current GPA: {str(e)}")
            return 0.0
    
    def _get_completed_courses(self, student_id):
        """
        الحصول على المواد التي اجتازها الطالب
        """
        try:
            # الحصول على الطالب
            student = Student.query.get(student_id)
            if not student:
                return []
            
            # الحصول على معرف القسم
            department_id = student.DepartmentId
            
            # الحصول على جميع التسجيلات المكتملة للطالب (حالة "ناجح")
            enrollments = Enrollment.query.filter_by(
                StudentId=student_id,
                IsCompleted="ناجح"
            ).all()
            
            logger.debug(f"Found {len(enrollments)} completed enrollments for student {student_id}")
            
            completed_courses = []
            for enrollment in enrollments:
                # الحصول على المادة
                course = Course.query.get(enrollment.CourseId)
                if course:
                    completed_courses.append({
                        "id": course.Id,
                        "code": course.Code if hasattr(course, 'Code') else "",
                        "name": course.Name,
                        "credits": course.Credits if hasattr(course, 'Credits') else 0,
                        "type": self._get_course_type(course.Id, department_id),
                        "grade": enrollment.Grade if hasattr(enrollment, 'Grade') else "",
                        "semester": enrollment.Semester if hasattr(enrollment, 'Semester') else ""
                    })
            
            return completed_courses
        except Exception as e:
            logger.error(f"Error getting completed courses: {str(e)}")
            return []
    
    def _get_course_type(self, course_id, department_id):
        """
        تحديد نوع المادة (إجباري أو اختياري)
        """
        try:
            # البحث عن المادة في جدول CourseDepartment
            course_department = CourseDepartment.query.filter_by(
                CourseId=course_id,
                DepartmentId=department_id
            ).first()
            
            # إذا وجدنا المادة، نحدد نوعها
            if course_department and hasattr(course_department, 'IsMandatory'):
                return "إجباري" if course_department.IsMandatory == 1 else "اختياري"
            
            # إذا لم نجد المادة، نفترض أنها إجبارية
            return "إجباري"
        except Exception as e:
            logger.error(f"Error getting course type: {str(e)}")
            return "إجباري"
    
    def _get_remaining_required_courses(self, student_id, department_id):
        """
        الحصول على المواد المتبقية المطلوبة للتخرج
        """
        try:
            # الحصول على المواد المكتملة
            completed_courses = self._get_completed_courses(student_id)
            completed_course_ids = [course['id'] for course in completed_courses]
            
            # الحصول على جميع المواد المطلوبة للقسم
            course_departments = CourseDepartment.query.filter_by(DepartmentId=department_id).all()
            
            remaining_courses = []
            for course_dept in course_departments:
                # تخطي المواد المكتملة
                if course_dept.CourseId in completed_course_ids:
                    continue
                
                # الحصول على المادة
                course = Course.query.get(course_dept.CourseId)
                if course:
                    remaining_courses.append({
                        "id": course.Id,
                        "code": course.Code if hasattr(course, 'Code') else "",
                        "name": course.Name,
                        "credits": course.Credits if hasattr(course, 'Credits') else 0,
                        "type": "إجباري" if course_dept.IsMandatory == 1 else "اختياري",
                        "description": course.Description if hasattr(course, 'Description') else ""
                    })
            
            return remaining_courses
        except Exception as e:
            logger.error(f"Error getting remaining required courses: {str(e)}")
            return []
    
    def _categorize_remaining_courses(self, remaining_courses):
        """
        تصنيف المواد المتبقية حسب النوع
        """
        try:
            categorized_courses = {
                "إجباري": [],
                "اختياري": []
            }
            
            for course in remaining_courses:
                course_type = course.get('type', 'إجباري')
                if course_type in categorized_courses:
                    categorized_courses[course_type].append(course)
            
            return categorized_courses
        except Exception as e:
            logger.error(f"Error categorizing remaining courses: {str(e)}")
            return {"إجباري": [], "اختياري": []}
    
    def _get_recommended_next_courses(self, student_id, remaining_courses, completed_courses, next_semester):
        """
        تحديد المواد الموصى بها للفصل الدراسي القادم
        """
        try:
            if not remaining_courses:
                return []
            
            # تحديد عدد المواد الموصى بها (5 مواد كحد أقصى)
            max_recommended_courses = 5
            
            # ترتيب المواد المتبقية حسب الأولوية (الإجبارية أولاً)
            sorted_courses = sorted(
                remaining_courses,
                key=lambda course: 0 if course.get('type') == "إجباري" else 1
            )
            
            # اختيار المواد الموصى بها
            recommended_courses = sorted_courses[:max_recommended_courses]
            
            return recommended_courses
        except Exception as e:
            logger.error(f"Error getting recommended next courses: {str(e)}")
            return []
    
    def _check_gpa_status(self, gpa):
        """
        تحديد حالة المعدل التراكمي
        """
        try:
            if gpa >= 3.5:
                return "ممتاز", "المعدل التراكمي ممتاز. استمر في العمل الجيد!"
            elif gpa >= 3.0:
                return "جيد جدًا", "المعدل التراكمي جيد جدًا. حافظ على هذا المستوى!"
            elif gpa >= 2.5:
                return "جيد", "المعدل التراكمي جيد. يمكنك تحسينه بمزيد من الجهد."
            elif gpa >= 2.0:
                return "مقبول", "المعدل التراكمي مقبول. يجب العمل على تحسينه."
            else:
                return "ضعيف", "المعدل التراكمي منخفض. يجب رفعه لتجنب الإنذار الأكاديمي."
        except Exception as e:
            logger.error(f"Error checking GPA status: {str(e)}")
            return "غير معروف", "لا يمكن تحديد حالة المعدل التراكمي."
    
    def _create_recommendations(self, student, gpa, completion_percentage, remaining_credits):
        """
        إنشاء توصيات للطالب
        """
        try:
            recommendations = []
            
            # توصيات بناءً على المعدل التراكمي
            if gpa < 2.0:
                recommendations.append({
                    "type": "تحذير",
                    "message": "المعدل التراكمي منخفض. يجب رفعه لتجنب الإنذار الأكاديمي."
                })
            elif gpa < 2.5:
                recommendations.append({
                    "type": "تنبيه",
                    "message": "المعدل التراكمي مقبول. يجب العمل على تحسينه."
                })
            
            # توصيات بناءً على نسبة الإنجاز
            if completion_percentage < 25:
                recommendations.append({
                    "type": "معلومة",
                    "message": "أنت في بداية الخطة الدراسية. ركز على المواد الأساسية."
                })
            elif completion_percentage < 50:
                recommendations.append({
                    "type": "معلومة",
                    "message": "أنت في منتصف الخطة الدراسية. حاول التركيز على المواد التخصصية."
                })
            elif completion_percentage < 75:
                recommendations.append({
                    "type": "معلومة",
                    "message": "أنت قريب من إكمال الخطة الدراسية. ركز على المواد المتبقية."
                })
            else:
                recommendations.append({
                    "type": "معلومة",
                    "message": "أنت على وشك التخرج. تأكد من إكمال جميع المتطلبات."
                })
            
            # توصيات بناءً على عدد الساعات المتبقية
            if remaining_credits > 30:
                recommendations.append({
                    "type": "نصيحة",
                    "message": f"لديك {remaining_credits} ساعة متبقية. حاول تسجيل الحد الأقصى من الساعات في كل فصل."
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error creating recommendations: {str(e)}")
            return []

class RecommendCoursesWithCredits(Resource):
    def get(self, student_id):
        try:
            # الحصول على بيانات الطالب
            student_data = get_student_data(student_id)
            if not student_data:
                return {"error": "Student not found"}, 404

            # الحصول على المواد المتاحة للفصل الدراسي الحالي وقسم الطالب
            available_courses = get_available_courses(student_data["current_semester"], student_data["department_id"])

            # الحصول على المتطلبات السابقة وبيانات المواد
            prerequisites = get_prerequisites()
            course_data = get_course_data()

            # الحصول على المواد الموصى بها
            result = recommend_courses(student_data, available_courses, course_data, prerequisites)
            
            # التحقق من شكل البيانات
            if isinstance(result, dict) and 'recommendations' in result and isinstance(result['recommendations'], list):
                # إضافة عدد الساعات لكل مادة
                for course in result['recommendations']:
                    if isinstance(course, dict) and 'id' in course:
                        # الحصول على المادة من قاعدة البيانات
                        db_course = Course.query.get(course['id'])
                        if db_course:
                            # إضافة عدد الساعات للمادة
                            course['credits'] = db_course.Credits
            
            return result, 200
            
        except Exception as e:
            logger.error(f"Error in RecommendCoursesWithCredits: {str(e)}")
            return {"error": str(e)}, 500

def get_current_enrolled_credits(student_id, semester_name):
    """
    حساب عدد الساعات المسجلة حاليًا للطالب في الفصل الدراسي المحدد
    """
    try:
        # الحصول على جميع التسجيلات النشطة للطالب في الفصل الدراسي المحدد
        enrollments = Enrollment.query.filter_by(
            StudentId=student_id,
            Semester=semester_name,
            DeletedEnrollmentDate=None,  # فقط التسجيلات النشطة
            IsCompleted="قيد الدراسة"    # فقط المواد قيد الدراسة
        ).all()
        
        # طباعة معلومات تصحيح
        logger.debug(f"Found {len(enrollments)} active enrollments for student {student_id} in semester {semester_name}")
        
        # حساب مجموع الساعات
        total_credits = 0
        for enrollment in enrollments:
            course = Course.query.get(enrollment.CourseId)
            if course and hasattr(course, 'Credits'):
                total_credits += course.Credits
                logger.debug(f"Course {course.Id} ({course.Name}) has {course.Credits} credits")
        
        logger.debug(f"Total credits for student {student_id} in semester {semester_name}: {total_credits}")
        return total_credits
    except Exception as e:
        logger.error(f"Error calculating current enrolled credits: {str(e)}")
        return 0  # في حالة حدوث خطأ، نفترض أن الطالب لم يسجل أي ساعات

def get_max_credits(student):
    """
    تحديد الحد الأقصى للساعات المسموح بها بناءً على المعدل التراكمي
    """
    try:
        # التحقق من وجود الطالب
        if student is None:
            logger.warning("Student is None, using default max credits (18)")
            return 18
        
        # الحصول على الفصل الدراسي الحالي للطالب
        current_semester = getattr(student, 'Semester', 0)
        logger.debug(f"Current semester for student: {current_semester}")
        
        # التحقق من أن الفصل الدراسي ضمن النطاق المسموح به
        if current_semester < 1 or current_semester > 8:
            logger.warning(f"Invalid semester: {current_semester}, using default max credits (18)")
            return 18
        
        # الحصول على المعدل التراكمي المناسب للفصل الدراسي
        gpa_field = f"GPA{current_semester}"
        gpa = getattr(student, gpa_field, None)
        logger.debug(f"GPA for semester {current_semester} ({gpa_field}): {gpa}")
        
        # التحقق من وجود المعدل
        if gpa is None:
            logger.warning(f"GPA for semester {current_semester} is None, using default max credits (18)")
            return 18
        
        # تحديد الحد الأقصى للساعات بناءً على المعدل
        if gpa < 2.0:  # ضعيف
            max_credits = 10
        else:  # 2.0 فأكثر
            max_credits = 18
        
        logger.debug(f"Max credits for GPA {gpa}: {max_credits}")
        return max_credits
    except Exception as e:
        logger.error(f"Error calculating max credits: {str(e)}")
        return 18  # قيمة افتراضية في حالة حدوث خطأ

class AcademicPerformanceEvaluation(Resource):
    def get(self, student_id):
        try:
            # التحقق من وجود الطالب
            student = Student.query.get(student_id)
            if not student:
                return {"error": "الطالب غير موجود"}, 404
            
            # الحصول على بيانات الطالب
            student_data = self._get_student_data(student)
            
            # تقييم الأداء الأكاديمي
            evaluation_report = self._evaluate_academic_performance(student_data)
            
            # تخزين التقرير في Redis لتحسين الأداء (اختياري)
            cache_key = f"academic_evaluation:{student_id}"
            redis_client.setex(cache_key, 3600, json.dumps(evaluation_report))  # تخزين لمدة ساعة
            
            return evaluation_report, 200
            
        except Exception as e:
            logger.error(f"Error in academic performance evaluation: {str(e)}")
            return {"error": f"حدث خطأ: {str(e)}"}, 500
    
    def _get_student_data(self, student):
        """جمع البيانات اللازمة لتقييم الأداء الأكاديمي للطالب"""
        
        # الحصول على المعدل التراكمي الحالي
        current_semester = student.Semester
        current_gpa_field = f'GPA{current_semester}'
        current_gpa = 0.0
        
        # محاولة الحصول على المعدل التراكمي للفصل الحالي
        if hasattr(student, current_gpa_field) and getattr(student, current_gpa_field) is not None:
            current_gpa = float(getattr(student, current_gpa_field))
        else:
            # إذا كان المعدل الحالي غير متوفر، استخدم آخر معدل متاح
            for i in range(current_semester - 1, 0, -1):
                prev_gpa_field = f'GPA{i}'
                if hasattr(student, prev_gpa_field) and getattr(student, prev_gpa_field) is not None:
                    current_gpa = float(getattr(student, prev_gpa_field))
                    break
        
        # الحصول على تاريخ المعدل التراكمي من الفصول السابقة
        gpa_history = []
        for i in range(1, current_semester + 1):
            gpa_field = f'GPA{i}'
            if hasattr(student, gpa_field) and getattr(student, gpa_field) is not None:
                gpa_history.append(float(getattr(student, gpa_field)))
        
        # الحصول على بيانات المواد والدرجات
        subjects = {}
        enrollments = Enrollment.query.filter_by(StudentId=student.Id).all()
        
        for enrollment in enrollments:
            course = Course.query.get(enrollment.CourseId)
            if course and hasattr(enrollment, 'Grade') and enrollment.Grade is not None:
                # حساب متوسط درجات الطلاب في هذه المادة
                avg_grade = db.session.query(func.avg(Enrollment.Grade)).filter(
                    Enrollment.CourseId == course.Id,
                    Enrollment.Grade.isnot(None)
                ).scalar() or 0
                
                subjects[course.Name] = {
                    "grade": float(enrollment.Grade),
                    "average_grade": float(avg_grade),
                    "credits": course.Credits
                }
        
        # الحصول على بيانات الحضور
        absences = {}
        attendance_records = Attendance.query.filter_by(StudentId=student.Id).all()
        
        for record in attendance_records:
            class_info = Class.query.get(record.ClassesId)
            if class_info:
                course = Course.query.get(class_info.CourseId)
                if course:
                    if course.Name not in absences:
                        absences[course.Name] = 0
                    
                    if not record.Status:  # Status = False يعني غائب
                        absences[course.Name] += 1
        
        # الحصول على عدد المواد التي رسب فيها الطالب
        failed_courses = db.session.query(func.count(Enrollment.Id)).filter(
            Enrollment.StudentId == student.Id,
            Enrollment.Grade < 60,
            Enrollment.IsCompleted == "راسب"
        ).scalar() or 0

        # الحصول على بيانات التدريب من قاعدة البيانات
        training_data = self._get_training_data_from_db()
        
        return {
            "student_id": student.Id,
            "name": student.Name,
            "current_gpa": current_gpa,
            "gpa_history": gpa_history,
            "subjects": subjects,
            "absences": absences,
            "failed_courses": failed_courses,
            "credits_completed": student.CreditsCompleted if hasattr(student, 'CreditsCompleted') else 0,
            "semester": student.Semester,
            "department_id": student.DepartmentId,
            "training_data": training_data
        }
    
    def _get_training_data_from_db(self):
        """الحصول على بيانات تدريب للنموذج التنبؤي من قاعدة البيانات"""
        cache_key = "training_data_for_academic_risk"
        cached_data = redis_client.get(cache_key)
        
        if cached_data:
            try:
                # تحويل البيانات المخزنة إلى DataFrame
                return pd.read_json(cached_data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Error reading cached training data: {str(e)}")
                # حذف البيانات المخزنة إذا كانت غير صالحة
                redis_client.delete(cache_key)
        
        # جمع بيانات من قاعدة البيانات لتدريب النموذج
        students = Student.query.all()
        
        # إعداد قوائم لتخزين البيانات
        training_data = []
        
        for student in students:
            try:
                # تخطي الطالب الحالي إذا كان هو نفسه الذي نقوم بتقييمه
                if hasattr(self, 'current_student_id') and student.Id == self.current_student_id:
                    continue
                    
                # الحصول على المعدل التراكمي الحالي
                current_semester = student.Semester
                current_gpa_field = f'GPA{current_semester}'
                
                if hasattr(student, current_gpa_field) and getattr(student, current_gpa_field) is not None:
                    current_gpa = float(getattr(student, current_gpa_field))
                else:
                    continue  # تخطي الطلاب بدون معدل تراكمي
                
                # حساب عدد الغيابات
                absence_count = db.session.query(func.count(Attendance.Id)).filter(
                    Attendance.StudentId == student.Id,
                    Attendance.Status == False
                ).scalar() or 0
                
                # حساب عدد المواد التي رسب فيها
                failed_enrollments = Enrollment.query.filter(
                    Enrollment.StudentId == student.Id,
                    Enrollment.IsCompleted == "راسب"
                ).all()
                
                failed_count = len(failed_enrollments)
                
                # تحديد ما إذا كان الطالب معرض للخطر
                at_risk = 1 if (current_gpa < 2.0 or failed_count > 2 or absence_count > 15) else 0
                
                # إضافة البيانات إلى القائمة
                training_data.append({
                    "gpa": current_gpa,
                    "absence": absence_count,
                    "failed_courses": failed_count,
                    "at_risk": at_risk
                })
                
            except Exception as e:
                logger.error(f"Error processing student {student.Id}: {str(e)}")
                continue
        
        # إنشاء DataFrame من البيانات المجمعة
        if training_data:
            df = pd.DataFrame(training_data)
            
            try:
                # تخزين البيانات في Redis
                redis_client.setex(
                    cache_key,
                    3600,  # تخزين لمدة ساعة
                    df.to_json().encode('utf-8')
                )
            except Exception as e:
                logger.error(f"Error caching training data: {str(e)}")
            
            return df
        else:
            logger.warning("No sufficient training data found")
            return None
    
    def _evaluate_academic_performance(self, student_data):
        """تقييم الأداء الأكاديمي للطالب"""
        
        # تحليل المعدل التراكمي
        gpa_analysis = self._gpa_analysis(student_data)
        
        # تحليل أداء المواد
        subject_performance = self._subject_performance(student_data)
        
        # تحليل الغيابح
        absence_analysis = self._absence_analysis(student_data)
        
        # الحصول على بيانات التدريب
        training_data = student_data.get("training_data")
        
        # تقييم المخاطر
        risk_assessment = self._risk_assessment(student_data, training_data)
        
        # الحصول على معلومات القسم
        department = Department.query.get(student_data.get("department_id"))
        department_name = department.Name if department else "غير محدد"
        
        # إنشاء التوصيات
        recommendations = self._generate_recommendations(
            student_data, 
            gpa_analysis, 
            subject_performance, 
            absence_analysis, 
            risk_assessment
        )
        
        # إنشاء التقرير النهائي
        report = {
            "student_info": {
                "id": student_data["student_id"],
                "name": student_data["name"],
                "department": department_name,
                "semester": student_data["semester"],
                "credits_completed": student_data["credits_completed"]
            },
            "gpa_analysis": gpa_analysis,
            "subject_performance": subject_performance,
            "absence_analysis": absence_analysis,
            "risk_assessment": risk_assessment,
            "recommendations": recommendations
        }
        
        return report

    def _generate_recommendations(self, student_data, gpa_analysis, subject_performance, 
                                 absence_analysis, risk_assessment):
        """توليد توصيات بناءً على تحليل البيانات"""
        recommendations = []
        
        # توصيات بناءً على المعدل التراكمي
        if gpa_analysis["status"] == "في خطر":
            recommendations.append("التركيز على تحسين المعدل التراكمي عبر مراجعة المواد الضعيفة")
        
        # توصيات بناءً على أداء المواد
        weak_subjects = [s["subject"] for s in subject_performance if "ضعيف" in s["performance_status"]]
        if weak_subjects:
            recommendations.append(f"مراجعة المواد التالية بسبب الأداء الضعيف: {', '.join(weak_subjects)}")
        
        # توصيات بناءً على الغياب
        critical_absences = absence_analysis.get("critical_subjects", [])
        if critical_absences:
            recommendations.append(f"تقليل الغياب خاصةً في المواد: {', '.join(critical_absences)}")
        
        # توصيات بناءً على تقييم المخاطر
        if risk_assessment["status"] == "الطالب معرض للخطر":
            recommendations.append("استشارة أكاديمية للحد من المخاطر الأكاديمية")
        
        # توصيات بناءً على عدد الساعات المكتملة
        credits_completed = student_data.get("credits_completed", 0)
        semester = student_data.get("semester", 1)
        expected_credits = semester * 18  # افتراض أن الطالب يكمل 18 ساعة في الفصل الواحد
        
        if credits_completed < expected_credits - 10:
            recommendations.append("زيادة عدد الساعات المسجلة في الفصول القادمة للتخرج في الوقت المحدد")
        
        return list(set(recommendations))

    @classmethod
    def _get_or_train_classifier(cls, training_data):
        """الحصول على نموذج مدرب أو تدريب نموذج جديد"""
        if training_data is None:
            return None
        
        try:
            # تدريب نموذج جديد
            classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            X_train = training_data[["gpa", "absence", "failed_courses"]]
            y_train = training_data["at_risk"]
            classifier.fit(X_train, y_train)
            
            return classifier
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            return None

    def _gpa_analysis(self, student_data):
        """تحليل المعدل التراكمي للطالب"""
        current_gpa = student_data["current_gpa"]
        gpa_history = student_data["gpa_history"]
        
        # تحليل اتجاه المعدل التراكمي
        gpa_trend = "مستقر"
        if len(gpa_history) >= 2:
            if gpa_history[-1] > gpa_history[-2]:
                gpa_trend = "تحسن"
            elif gpa_history[-1] < gpa_history[-2]:
                gpa_trend = "تراجع"
        
        # تحديد حالة المعدل التراكمي
        if current_gpa >= 3.5:
            status = "ممتاز"
        elif current_gpa >= 3.0:
            status = "جيد جداً"
        elif current_gpa >= 2.5:
            status = "جيد"
        elif current_gpa >= 2.0:
            status = "مقبول"
        else:
            status = "في خطر"
        
        return {
            "current_gpa": current_gpa,
            "gpa_history": gpa_history,
            "trend": gpa_trend,
            "status": status
        }

    def _subject_performance(self, student_data):
        """تحليل أداء المواد"""
        subjects = student_data["subjects"]
        subject_analysis = []
        
        for subject_name, data in subjects.items():
            # تحليل الأداء مقارنة بمتوسط الفصل
            performance_diff = data["grade"] - data["average_grade"]
            
            # تحديد حالة الأداء بناءً على الدرجة النهائية (من 90)
            if data["grade"] >= 81:  # 90% من 90
                performance_status = "ممتاز"
            elif data["grade"] >= 72:  # 80% من 90
                performance_status = "جيد جداً"
            elif data["grade"] >= 63:  # 70% من 90
                performance_status = "جيد"
            elif data["grade"] >= 54:  # 60% من 90
                performance_status = "مقبول"
            else:
                performance_status = "ضعيف"
            
            subject_analysis.append({
                "subject": subject_name,
                "grade": data["grade"],
                "class_average": data["average_grade"],
                "performance_status": performance_status,
                "difference_from_average": round(performance_diff, 2)
            })
        
        return subject_analysis

    def _absence_analysis(self, student_data):
        """تحليل الغياب"""
        absences = student_data["absences"]
        critical_subjects = []
        total_absences = 0
        
        # الحصول على معلومات الفصل الدراسي الحالي
        current_semester = student_data.get("semester", 1)
        current_week = self._get_current_week_in_semester()  # دالة جديدة لتحديد الأسبوع الحالي
        
        # الحصول على معلومات المواد
        courses = {}
        for subject_name in absences.keys():
            course = Course.query.filter_by(Name=subject_name).first()
            if course:
                courses[subject_name] = {
                    "credits": course.Credits,
                    "lectures_per_week": course.LecturesPerWeek if hasattr(course, 'LecturesPerWeek') else 1  # افتراضي محاضرة واحدة أسبوعيًا
                }
        
        # حساب نسبة الغياب لكل مادة
        absence_percentages = {}
        for subject_name, count in absences.items():
            total_absences += count
            
            # حساب عدد المحاضرات حتى الآن في الفصل الدراسي
            if subject_name in courses:
                lectures_per_week = courses[subject_name]["lectures_per_week"]
                lectures_so_far = lectures_per_week * current_week
                
                # حساب نسبة الغياب بناءً على المحاضرات حتى الآن
                if lectures_so_far > 0:
                    absence_percentage = round((count / lectures_so_far) * 100, 1)  # تقريب إلى رقم عشري واحد
                else:
                    absence_percentage = 0
                    
                absence_percentages[subject_name] = absence_percentage
                
                # تحديد المواد ذات نسبة الغياب العالية
                if absence_percentage > 25:  # أكثر من 25% غياب يعتبر حرجًا
                    critical_subjects.append(subject_name)
            else:
                # إذا لم تكن معلومات المادة متاحة، استخدم العدد المطلق
                if count > 3:  # أكثر من 3 غيابات يعتبر حرجًا
                    critical_subjects.append(subject_name)
        
        # تحديد حالة الغياب الإجمالية
        avg_absence_percentage = sum(absence_percentages.values()) / len(absence_percentages) if absence_percentages else 0
        
        if avg_absence_percentage > 30:
            absence_status = "مرتفع جدًا"
        elif avg_absence_percentage > 20:
            absence_status = "مرتفع"
        elif avg_absence_percentage > 10:
            absence_status = "متوسط"
        else:
            absence_status = "طبيعي"
        
        return {
            "total_absences": total_absences,
            "critical_subjects": critical_subjects,
            "absence_percentages": absence_percentages,
            "absence_status": absence_status,
            "current_week": current_week  # إضافة الأسبوع الحالي للمعلومات
        }

    def _get_current_week_in_semester(self):
        """تحديد الأسبوع الحالي في الفصل الدراسي"""
        # استخدام تقدير بناءً على الشهر الحالي فقط
        current_month = datetime.now().month
        
        # افتراض أن الفصل الأول يبدأ في سبتمبر والفصل الثاني في فبراير
        if 9 <= current_month <= 12:  # الفصل الأول (سبتمبر - ديسمبر)
            week_estimate = (current_month - 9) * 4 + min(datetime.now().day // 7 + 1, 4)
        elif 2 <= current_month <= 5:  # الفصل الثاني (فبراير - مايو)
            week_estimate = (current_month - 2) * 4 + min(datetime.now().day // 7 + 1, 4)
        else:
            # خارج فترة الفصل الدراسي العادية
            week_estimate = 7  # افتراض منتصف الفصل
        
        return min(max(week_estimate, 1), 14)  # بين 1 و 14

    def _risk_assessment(self, student_data, training_data):
        """تقييم مخاطر الأداء الأكاديمي"""
        try:
            # استخدام آخر معدل تراكمي متاح بدلاً من المعدل الحالي إذا كان 0
            current_gpa = student_data["current_gpa"]
            if current_gpa == 0.0 and student_data["gpa_history"]:
                current_gpa = student_data["gpa_history"][-1]
            
            # التحقق من وجود بيانات التدريب
            if training_data is None or len(training_data) < 10:  # نحتاج على الأقل 10 طلاب للتدريب
                # حساب عوامل الخطر
                low_gpa = current_gpa < 2.0
                
                # تحسين تقييم الغياب
                absence_analysis = student_data.get("absence_analysis", {})
                absence_status = absence_analysis.get("absence_status", "طبيعي")
                high_absence = absence_status in ["مرتفع", "مرتفع جدًا"]
                
                # أو بديلاً، يمكن استخدام النسبة المئوية للغياب
                absence_percentages = absence_analysis.get("absence_percentages", {})
                avg_absence_percentage = sum(absence_percentages.values()) / len(absence_percentages) if absence_percentages else 0
                high_absence = avg_absence_percentage > 20  # أكثر من 20% غياب في المتوسط
                
                failed_courses = student_data["failed_courses"] > 2
                
                # حساب عدد عوامل الخطر
                risk_factors_count = sum([low_gpa, high_absence, failed_courses])
                
                # حساب احتمالية الخطر بناءً على عدد عوامل الخطر
                if risk_factors_count == 0:
                    risk_probability = 0.1  # خطر منخفض جداً
                elif risk_factors_count == 1:
                    risk_probability = 0.4  # خطر منخفض إلى متوسط
                elif risk_factors_count == 2:
                    risk_probability = 0.7  # خطر متوسط إلى مرتفع
                else:  # 3 عوامل خطر
                    risk_probability = 0.9  # خطر مرتفع جداً
                
                # تحديد حالة الخطر
                risk_status = "الطالب معرض للخطر" if risk_probability > 0.5 else "الطالب في وضع جيد"
                
                # تعديل أهمية المتغيرات بناءً على حالة الطالب
                if low_gpa:
                    # إذا كان المعدل منخفضًا، زيادة أهمية المعدل
                    feature_importance = {
                        "gpa": 0.7,
                        "failed_courses": 0.2,
                        "absence": 0.1
                    }
                elif high_absence:
                    # إذا كان الغياب مرتفعًا، زيادة أهمية الغياب
                    feature_importance = {
                        "gpa": 0.5,
                        "failed_courses": 0.2,
                        "absence": 0.3
                    }
                elif failed_courses:
                    # إذا كان هناك مواد راسب فيها، زيادة أهمية المواد الراسبة
                    feature_importance = {
                        "gpa": 0.5,
                        "failed_courses": 0.4,
                        "absence": 0.1
                    }
                else:
                    # الحالة الافتراضية
                    feature_importance = {
                        "gpa": 0.6,
                        "failed_courses": 0.25,
                        "absence": 0.15
                    }
                
                return {
                    "status": risk_status,
                    "probability": risk_probability,
                    "factors": {
                        "low_gpa": low_gpa,
                        "high_absence": high_absence,
                        "failed_courses": failed_courses
                    },
                    "feature_importance": feature_importance
                }
            
            # تدريب  
            classifier = self._get_or_train_classifier(training_data)
            
            # إعداد بيانات الطالب للتنبؤ
            student_features = pd.DataFrame({
                "gpa": [current_gpa],
                "absence": [sum(student_data["absences"].values())],
                "failed_courses": [student_data["failed_courses"]]
            })
            
            # التنبؤ باحتمالية المخاطر
            risk_proba = classifier.predict_proba(student_features)
            if risk_proba.shape[1] < 2:
                raise ValueError("خطأ في تنسيق احتمالات التنبؤ")
                
            risk_probability = risk_proba[0][1]  # احتمالية أن يكون الطالب معرض للخطر
            risk_status = "الطالب معرض للخطر" if risk_probability > 0.5 else "الطالب في وضع جيد"
            
            # حساب أهمية المتغيرات
            feature_importance = dict(zip(
                ["gpa", "absence", "failed_courses"],
                classifier.feature_importances_
            ))
            
            return {
                "status": risk_status,
                "probability": float(risk_probability),  # تحويل إلى float للتأكد من إمكانية التحويل إلى JSON
                "factors": {
                    "low_gpa": current_gpa < 2.0,
                    "high_absence": sum(student_data["absences"].values()) > 15,
                    "failed_courses": student_data["failed_courses"] > 2
                },
                "feature_importance": {k: float(v) for k, v in feature_importance.items()}  # تحويل القيم إلى float
            }
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            
            # في حالة حدوث خطأ، استخدم تقييم بسيط بناءً على القواعد
            current_gpa = student_data["current_gpa"]
            if current_gpa == 0.0 and student_data["gpa_history"]:
                current_gpa = student_data["gpa_history"][-1]
                
            low_gpa = current_gpa < 2.0
            high_absence = sum(student_data["absences"].values()) > 15
            failed_courses = student_data["failed_courses"] > 2
            
            # تحديد حالة الخطر بناءً على عوامل الخطر
            is_at_risk = low_gpa or high_absence or failed_courses
            
            return {
                "status": "الطالب معرض للخطر" if is_at_risk else "الطالب في وضع جيد",
                "probability": 0.8 if is_at_risk else 0.2,
                "factors": {
                    "low_gpa": low_gpa,
                    "high_absence": high_absence,
                    "failed_courses": failed_courses
                },
                "feature_importance": {
                    "gpa": 0.6,
                    "failed_courses": 0.25,
                    "absence": 0.15
                }
            }

class GraduationCheckResource(Resource):
    def get(self, student_id):
        """التحقق من استيفاء متطلبات التخرج للطالب"""
        try:
            # الحصول على بيانات الطالب
            student = Student.query.get(student_id)
            if not student:
                return {"error": "الطالب غير موجود"}, 404
            
            # الحصول على متطلبات التخرج للقسم
            department = Department.query.get(student.DepartmentId)
            if not department:
                return {"error": "القسم غير موجود"}, 404
            
            # الحصول على الساعات المعتمدة المطلوبة للتخرج
            required_credits = department.RequiredCredits if hasattr(department, 'RequiredCredits') else 136
            
            # حساب الساعات المعتمدة المكتملة
            completed_credits = student.CreditsCompleted if hasattr(student, 'CreditsCompleted') else 0
            
            # حساب الساعات المتبقية
            remaining_credits = required_credits - completed_credits
            
            # الحصول على المواد الإلزامية للقسم
            mandatory_courses = CourseDepartment.query.filter_by(
                DepartmentId=student.DepartmentId,
                IsMandatory=True
            ).all()
            
            total_mandatory_courses = len(mandatory_courses)
            
            # التحقق من المواد الإلزامية المكتملة
            completed_mandatory_courses = 0
            incomplete_mandatory_courses = []
            
            for course_dept in mandatory_courses:
                course = Course.query.get(course_dept.CourseId)
                if not course:
                    continue
                
                # التحقق مما إذا كان الطالب قد أكمل هذه المادة
                enrollment = Enrollment.query.filter_by(
                    StudentId=student_id,
                    CourseId=course.Id,
                    IsCompleted="ناجح"
                ).first()
                
                if enrollment:
                    completed_mandatory_courses += 1
                else:
                    incomplete_mandatory_courses.append(course.Name)
            
            # تحديد ما إذا كان الطالب مؤهلاً للتخرج
            is_eligible = (completed_credits >= required_credits and 
                          completed_mandatory_courses == total_mandatory_courses)
            
            # إعداد قائمة بأسباب عدم الأهلية
            reasons = []
            if completed_credits < required_credits:
                reasons.append(f"الساعات المكتملة أقل من {required_credits} (متبقي {remaining_credits} ساعة)")
            
            if completed_mandatory_courses < total_mandatory_courses:
                reasons.append(f"توجد {total_mandatory_courses - completed_mandatory_courses} مواد إلزامية غير مكتملة")
                if len(incomplete_mandatory_courses) > 0:
                    # إضافة قائمة بالمواد الإلزامية غير المكتملة (بحد أقصى 5 مواد)
                    courses_to_show = incomplete_mandatory_courses[:5]
                    if len(incomplete_mandatory_courses) > 5:
                        courses_to_show.append("... وغيرها")
                    
                    reasons.append(f"المواد الإلزامية غير المكتملة: {', '.join(courses_to_show)}")
            
            
            print(f"Student: {student.Name}, Remaining Credits: {remaining_credits}")
            
            response = {
                "student_id": student_id,
                "student_name": student.Name,
                "total_credits": completed_credits,
                "required_credits": required_credits,
                "remaining_credits": remaining_credits,
                "completed_mandatory_courses": completed_mandatory_courses,
                "total_mandatory_courses": total_mandatory_courses,
                "is_eligible": is_eligible,
                "reasons": reasons
            }
            
           
            if incomplete_mandatory_courses:
                response["incomplete_mandatory_courses"] = incomplete_mandatory_courses[:10]
                if len(incomplete_mandatory_courses) > 10:
                    response["incomplete_mandatory_courses"].append("... وغيرها")
            
            return response
            
        except Exception as e:
            print(f"Error in graduation check: {str(e)}")
            return {"error": f"حدث خطأ: {str(e)}"}, 500