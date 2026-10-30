
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

project_folder = Path(__file__).resolve().parent
csv_path = project_folder / "student_marks.csv"
graph_folder = project_folder / "graphs"

subjects = ["Python", "DBMS", "Mathematics"]
pass_marks = 40
maximum_marks = 100

print("=" * 60)
print("       STUDENT PERFORMANCE ANALYSIS SYSTEM")
print("=" * 60)

print("\nProject folder:", project_folder)
print("CSV file:", csv_path)


# ============================================================
# 2. LOAD AND VALIDATE STUDENT DATA
# ============================================================

if not csv_path.exists():
    print("\nERROR: student_marks.csv was not found.")
    print("Keep student_marks.csv in the same folder as main.py.")
    raise SystemExit(1)

try:
    # Load the three subject columns from the CSV file.
    marks = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3),
        dtype=float
    )

    # Ensure the data is always a 2D array.
    marks = np.atleast_2d(marks)

except (ValueError, IndexError) as error:
    print("\nERROR: Could not read the CSV marks.")
    print("Check that the CSV contains numeric marks.")
    print("Details:", error)
    raise SystemExit(1)

# Check that marks are within the expected range.
if not np.all(np.isfinite(marks)):
    print("ERROR: The marks contain invalid numeric values.")
    raise SystemExit(1)

if np.any((marks < 0) | (marks > maximum_marks)):
    print("ERROR: Marks must be between 0 and 100.")
    raise SystemExit(1)

number_of_students = marks.shape[0]
number_of_subjects = marks.shape[1]

if number_of_subjects != len(subjects):
    print("ERROR: Expected exactly three subject columns.")
    raise SystemExit(1)

# Load student names from the first CSV column.
try:
    names = np.genfromtxt(
        csv_path,
        delimiter=",",
        skip_header=1,
        usecols=0,
        dtype=str,
        encoding="utf-8"
    )
    names = np.atleast_1d(names)

except (ValueError, IndexError) as error:
    print("ERROR: Could not read student names:", error)
    raise SystemExit(1)

if len(names) != number_of_students:
    print("ERROR: Student names and marks have different lengths.")
    raise SystemExit(1)

print("\nData loaded successfully!")
print("Total students:", number_of_students)
print("Total subjects:", number_of_subjects)


# ============================================================
# 3. BASIC STUDENT CALCULATIONS
# ============================================================

# Calculate each student's total marks.
total_marks = np.sum(marks, axis=1)

# Calculate percentage out of 100.
percentage = np.mean(marks, axis=1)

# Calculate each student's average marks.
average_marks = np.mean(marks, axis=1)

# Find the highest and lowest marks for each student.
highest_student_mark = np.max(marks, axis=1)
lowest_student_mark = np.min(marks, axis=1)

# Count subjects passed by each student.
subjects_passed = np.sum(marks >= pass_marks, axis=1)

# A student passes only if every subject has at least 40 marks.
student_passed = np.all(marks >= pass_marks, axis=1)


# ============================================================
# 4. GRADE CALCULATION
# ============================================================

def calculate_grade(score):
    """Return a grade based on the student's percentage."""

    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    elif score >= 40:
        return "E"
    else:
        return "F"


grades = np.array([
    calculate_grade(score)
    for score in percentage
])


# ============================================================
# 5. STUDENT RANKING
# ============================================================

# Highest percentage receives rank 1.
sorted_indices = np.argsort(-percentage)

ranks = np.empty(number_of_students, dtype=int)

for position, student_index in enumerate(sorted_indices, start=1):
    ranks[student_index] = position


# ============================================================
# 6. DISPLAY ALL STUDENT RESULTS
# ============================================================

print("\n" + "=" * 90)
print("STUDENT RESULT TABLE")
print("=" * 90)

print(
    f"{'Name':<18}"
    f"{'Python':>9}"
    f"{'DBMS':>9}"
    f"{'Maths':>9}"
    f"{'Total':>9}"
    f"{'Percent':>10}"
    f"{'Grade':>8}"
    f"{'Rank':>7}"
    f"{'Result':>10}"
)

print("-" * 90)

for i in range(number_of_students):
    result = "PASS" if student_passed[i] else "FAIL"

    print(
        f"{names[i]:<18}"
        f"{marks[i, 0]:>9.1f}"
        f"{marks[i, 1]:>9.1f}"
        f"{marks[i, 2]:>9.1f}"
        f"{total_marks[i]:>9.1f}"
        f"{percentage[i]:>10.2f}"
        f"{grades[i]:>8}"
        f"{ranks[i]:>7}"
        f"{result:>10}"
    )


# ============================================================
# 7. OVERALL CLASS STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("OVERALL CLASS STATISTICS")
print("=" * 60)

print(f"Number of students: {number_of_students}")
print(f"Number of subjects: {number_of_subjects}")
print(f"Class average: {np.mean(percentage):.2f}%")
print(f"Highest percentage: {np.max(percentage):.2f}%")
print(f"Lowest percentage: {np.min(percentage):.2f}%")
print(f"Median percentage: {np.median(percentage):.2f}%")
print(f"Percentage standard deviation: {np.std(percentage):.2f}")
print(f"Average total marks: {np.mean(total_marks):.2f}")
print(f"Maximum total marks: {np.max(total_marks):.2f}")
print(f"Minimum total marks: {np.min(total_marks):.2f}")


# ============================================================
# 8. SUBJECT-WISE ANALYSIS
# ============================================================

subject_averages = np.mean(marks, axis=0)
subject_highest = np.max(marks, axis=0)
subject_lowest = np.min(marks, axis=0)
subject_medians = np.median(marks, axis=0)
subject_std = np.std(marks, axis=0)

print("\n" + "=" * 60)
print("SUBJECT-WISE ANALYSIS")
print("=" * 60)

for i, subject in enumerate(subjects):
    print(f"\nSubject: {subject}")
    print(f"Average marks: {subject_averages[i]:.2f}")
    print(f"Highest marks: {subject_highest[i]:.2f}")
    print(f"Lowest marks: {subject_lowest[i]:.2f}")
    print(f"Median marks: {subject_medians[i]:.2f}")
    print(f"Standard deviation: {subject_std[i]:.2f}")


# ============================================================
# 9. FIND THE BEST AND WEAKEST SUBJECTS
# ============================================================

best_subject_index = np.argmax(subject_averages)
weakest_subject_index = np.argmin(subject_averages)

print("\n" + "=" * 60)
print("BEST AND WEAKEST SUBJECT")
print("=" * 60)

print(
    "Best-performing subject:",
    subjects[best_subject_index],
    f"({subject_averages[best_subject_index]:.2f} average)"
)

print(
    "Lowest-average subject:",
    subjects[weakest_subject_index],
    f"({subject_averages[weakest_subject_index]:.2f} average)"
)


# ============================================================
# 10. FIND THE TOP FIVE STUDENTS
# ============================================================

print("\n" + "=" * 60)
print("TOP FIVE STUDENTS")
print("=" * 60)

top_count = min(5, number_of_students)

for position in range(top_count):
    student_index = sorted_indices[position]

    print(
        f"Rank {position + 1}: {names[student_index]} "
        f"| Percentage: {percentage[student_index]:.2f}% "
        f"| Grade: {grades[student_index]}"
    )


# ============================================================
# 11. PASS AND FAIL ANALYSIS
# ============================================================

passed_count = np.sum(student_passed)
failed_count = number_of_students - passed_count

pass_percentage = (passed_count / number_of_students) * 100
fail_percentage = (failed_count / number_of_students) * 100

print("\n" + "=" * 60)
print("PASS AND FAIL ANALYSIS")
print("=" * 60)

print(f"Students passed: {passed_count}")
print(f"Students failed: {failed_count}")
print(f"Pass percentage: {pass_percentage:.2f}%")
print(f"Fail percentage: {fail_percentage:.2f}%")


# ============================================================
# 12. GRADE DISTRIBUTION
# ============================================================

grade_names = ["A+", "A", "B", "C", "D", "E", "F"]

grade_counts = np.array([
    np.sum(grades == grade)
    for grade in grade_names
])

print("\n" + "=" * 60)
print("GRADE DISTRIBUTION")
print("=" * 60)

for grade, count in zip(grade_names, grade_counts):
    print(f"Grade {grade}: {count} student(s)")


# ============================================================
# 13. FIND STUDENTS WHO NEED IMPROVEMENT
# ============================================================

print("\n" + "=" * 60)
print("STUDENTS WHO NEED IMPROVEMENT")
print("=" * 60)

weak_student_found = False

for i in range(number_of_students):
    weak_subjects = [
        subjects[j]
        for j in range(number_of_subjects)
        if marks[i, j] < pass_marks
    ]

    if weak_subjects:
        weak_student_found = True
        print(
            f"{names[i]} needs improvement in: "
            f"{', '.join(weak_subjects)}"
        )

if not weak_student_found:
    print("No student has scored below the pass mark in any subject.")


# ============================================================
# 14. SUBJECT-WISE PASS COUNTS
# ============================================================

print("\n" + "=" * 60)
print("SUBJECT-WISE PASS COUNTS")
print("=" * 60)

for i, subject in enumerate(subjects):
    subject_pass_count = np.sum(marks[:, i] >= pass_marks)
    subject_fail_count = number_of_students - subject_pass_count

    print(
        f"{subject}: "
        f"{subject_pass_count} passed, "
        f"{subject_fail_count} below pass mark"
    )


# ============================================================
# 15. CORRELATION BETWEEN SUBJECTS
# ============================================================

correlation_matrix = np.corrcoef(marks, rowvar=False)

print("\n" + "=" * 60)
print("CORRELATION BETWEEN SUBJECTS")
print("=" * 60)

print("Rows and columns follow this subject order:", subjects)
print(np.round(correlation_matrix, 2))

print(
    "\nCorrelation describes how marks vary together. "
    "It does not prove that one subject causes better marks in another."
)


# ============================================================
# 16. SAVE OUTPUT GRAPHS
# ============================================================

graph_folder.mkdir(exist_ok=True)

# Graph 1: Average marks by subject.
plt.figure(figsize=(9, 5))
bars = plt.bar(subjects, subject_averages)

plt.title("Average Marks by Subject")
plt.xlabel("Subjects")
plt.ylabel("Average Marks")
plt.ylim(0, 100)

for bar, value in zip(bars, subject_averages):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(graph_folder / "subject_average.png", dpi=150)
plt.show()


# Graph 2: Student percentages.
plt.figure(figsize=(12, 6))
plt.bar(names, percentage)

plt.title("Student Percentages")
plt.xlabel("Students")
plt.ylabel("Percentage")
plt.ylim(0, 100)
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig(graph_folder / "student_percentages.png", dpi=150)
plt.show()


# Graph 3: Grade distribution.
plt.figure(figsize=(8, 5))
plt.bar(grade_names, grade_counts)

plt.title("Grade Distribution")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(graph_folder / "grade_distribution.png", dpi=150)
plt.show()


# Graph 4: Pass and fail counts.
plt.figure(figsize=(7, 5))
plt.bar(["Passed", "Failed"], [passed_count, failed_count])

plt.title("Class Pass and Fail Analysis")
plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(graph_folder / "pass_fail.png", dpi=150)
plt.show()


# Graph 5: Subject comparison for all students.
plt.figure(figsize=(12, 6))

positions = np.arange(number_of_students)
width = 0.25

for subject_index, subject in enumerate(subjects):
    plt.bar(
        positions + subject_index * width,
        marks[:, subject_index],
        width,
        label=subject
    )

plt.title("Subject Marks Comparison")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.ylim(0, 100)

plt.xticks(
    positions + width,
    names,
    rotation=75,
    ha="right"
)

plt.legend()
plt.tight_layout()
plt.savefig(graph_folder / "subject_comparison.png", dpi=150)
plt.show()


# Graph 6: Marks distribution.
plt.figure(figsize=(9, 5))

for subject_index, subject in enumerate(subjects):
    plt.hist(
        marks[:, subject_index],
        bins=10,
        alpha=0.5,
        label=subject
    )

plt.title("Distribution of Subject Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.savefig(graph_folder / "marks_distribution.png", dpi=150)
plt.show()


# Graph 7: Subject correlation heatmap.
plt.figure(figsize=(7, 6))

image = plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(image, label="Correlation")
plt.xticks(range(len(subjects)), subjects, rotation=30)
plt.yticks(range(len(subjects)), subjects)
plt.title("Subject Marks Correlation")

for row in range(len(subjects)):
    for column in range(len(subjects)):
        value = correlation_matrix[row, column]

        if np.isfinite(value):
            plt.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center"
            )
        else:
            plt.text(
                column,
                row,
                "N/A",
                ha="center",
                va="center"
            )

plt.tight_layout()
plt.savefig(graph_folder / "subject_correlation.png", dpi=150)
plt.show()

# ============================================================
# 17. FINAL SUMMARY
# ============================================================

top_student_index = np.argmax(percentage)
bottom_student_index = np.argmin(percentage)

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print(f"Students analysed: {number_of_students}")
print(f"Subjects analysed: {number_of_subjects}")
print(f"Class average: {np.mean(percentage):.2f}%")

print(
    f"Top student: {names[top_student_index]} "
    f"({percentage[top_student_index]:.2f}%)"
)

print(
    f"Lowest percentage: {names[bottom_student_index]} "
    f"({percentage[bottom_student_index]:.2f}%)"
)

print(f"Overall pass rate: {pass_percentage:.2f}%")
print(f"Graphs saved in: {graph_folder}")

print("\nAnalysis completed successfully!")
print("You can use the saved graphs in your project report.")
