# sql1.py
"""Volume 1: SQL 1 (Introduction).
Sam Goldrup
MATH 347
22 March 2022
"""

from ast import AsyncFunctionDef
import sqlite3 as sql
import csv
from matplotlib import pyplot as plt
import numpy as np

# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    with open(student_info,'r') as infile: #open files with defualt names
        info_students = list(csv.reader(infile))

    with open(student_grades) as infile:
        grades_info = list(csv.reader(infile))

    major_info = [(1,"Math"),(2,"Science"),(3,"Writing"),(4,"Art")] #make up some data
    course_info = [(1,"Calculus"),(2,"English"),(3,"Pottery"),(4,"History")]

    try:
        with sql.connect(db_file) as conn: #connect to the file
            #problem 1
            cur = conn.cursor()
            cur.execute("DROP TABLE IF EXISTS MajorInfo") #drop tables
            cur.execute("DROP TABLE IF EXISTS CourseInfo")
            cur.execute("DROP TABLE IF EXISTS StudentInfo")
            cur.execute("DROP TABLE IF EXISTS StudentGrades")

            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT)") #make new tables with columns and their types
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT)")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER)")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT)")

            #problem 2
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);",major_info) #enter in the data
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);",course_info)
            cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);",info_students)
            cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);",grades_info)

            #problem 4
            cur.execute("UPDATE StudentInfo SET MajorID=NULL WHERE MajorID==-1;") #clean some data
    finally:
        conn.commit() #save changes
        conn.close() #x out

# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    with open(data_file) as infile: #read in the data to enter
        ass_quakes = list(csv.reader(infile))

    try:
        with sql.connect(db_file) as conn: #connect to the file
            cur = conn.cursor()
            cur.execute("DROP TABLE IF EXISTS USEarthquakes") #drop the table
            #aaaand make it again
            cur.execute("CREATE TABLE USEarthquakes (Year INTEGER,Month INTEGER,Day INTEGER,Hour INTEGER,Minute INTEGER,Second INTEGER,Latitude REAL,Longitude REAL,Magnitude REAL);")
            #put it in baby
            cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);",ass_quakes)

            #problem 4
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude == 0") #clean data
            cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day==0")
            cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour=0")
            cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute=0")
            cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second=0")
    finally:
        conn.commit() #save changes
        conn.close()



# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #use abbreviations, get the desired elements from them, matching on StudentID, CourseID and picking those with A or A+
            cur.execute("SELECT SI.StudentName, CI.CourseName "
                        "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades AS SG "
                        "WHERE SI.StudentID == SG.StudentID AND (SG.Grade == 'A+' OR SG.Grade=='A') AND CI.CourseID==SG.CourseID;")
            fetch = cur.fetchall()
    finally:
        conn.close() #x out of the file
            
    return fetch

# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    earthquakes_db() #assume its in the right format
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            mags_19 = cur.execute("SELECT USEQ.Magnitude " #19th century earthquakes
                                  "FROM USEarthQuakes AS USEQ "
                                  "WHERE USEQ.YEAR >= 1800 AND USEQ.YEAR <= 1899").fetchall()
            mags_19 = np.array(mags_19,dtype=float) #convert to an array
            mags_20 = cur.execute("SELECT USEQ.Magnitude " #20th cenutry earthquakes
                                  "FROM USEarthQuakes AS USEQ "
                                  "WHERE USEQ.YEAR >= 1900 AND USEQ.YEAR <= 1999").fetchall()
            mags_20 = np.array(mags_20,dtype=float)
            avg_mag = cur.execute("SELECT AVG(Magnitude) from USEarthquakes;").fetchall()[0][0] #its deep inside a tuple
    finally:
        conn.close()

    plt.subplot(121).hist(mags_19) #one for each century
    plt.title("19th century") #labels
    plt.xlabel("magnitude")
    plt.ylabel("number of quakes")
    plt.subplot(122).hist(mags_20) 
    plt.title("20th century")
    plt.xlabel("magnitude")
    plt.ylabel("number of quakes")
    plt.suptitle("proof of the end of world") #overall title
    plt.tight_layout() #make it look good
    plt.show()

    return avg_mag
    


if __name__ == "__main__":
    student_db()
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM StudentInfo")
        print([d[0] for d in cur.description])

        for row in cur.execute("SELECT * FROM MajorInfo;"):
            print(row)
        for row in cur.execute("SELECT * FROM CourseInfo;"):
            print(row)
        for row in cur.execute("SELECT * FROM StudentInfo;"):
            print(row)
        for row in cur.execute("SELECT * FROM StudentGrades;"):
            print(row)

    conn.close()

    earthquakes_db()
    with sql.connect("earthquakes.db") as conn:
        cur = conn.cursor()
        for row in cur.execute("SELECT * FROM USEarthquakes;"):
            print(row)
    conn.close()

    print(prob5())
    print(prob6())
