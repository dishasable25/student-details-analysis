import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier  # <-- Newly added

# Load the dataset
data = pd.read_csv("std_datas.csv")

# Fill empty strings only in object (text) columns
for col in data.select_dtypes(include='object').columns:
    data[col].fillna('', inplace=True)

# Manual mapping for Gender (make sure gender is "Male"/"Female" in CSV)
gender_map = {'Female': 0, 'Male': 1}
data['Gender'] = data['Gender'].map(gender_map)

# Convert numeric columns and handle non-numeric entries safely
numeric_columns = ['Age', 'CGPA', 'SGPA', 'Attendance (%)', 'Income (Yearly)']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Label encode Course, Branch, and Scholarship Status
label_encoders = {}
for col in ['Course', 'Branch', 'Scholarship Status']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le  # Store for potential reverse lookup

# Drop any rows with missing feature values or target
features = ['Age', 'Gender', 'CGPA', 'SGPA', 'Attendance (%)', 'Course', 'Branch']
target = 'Scholarship Status'
data.dropna(subset=features + [target], inplace=True)

# Train-test split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------- Newly Added: Multi-label Classifier -------------------
# Multi-label targets
multi_target = pd.DataFrame({
    'General Scholarship': (
        (data['Attendance (%)'] > 75) & 
        (data['CGPA'] > 6) & 
        (data['SGPA'] > 6)
    ),
    'Reliance Scholarship': (
        (data['CGPA'] > 7) & 
        (data['SGPA'] > 7) & 
        (data['Attendance (%)'] > 75)
    ),
    'PM Scholarship': (
        (data['Income (Yearly)'] < 50000) & 
        (data['CGPA'] > 6) & 
        (data['SGPA'] > 6) & 
        (data['Extracurricular'] != '')
    ),
    'Indira Gandhi Scholarship': (
        (data['Gender'] == 0) & 
        (data['CGPA'] > 6) & 
        (data['Extracurricular'] != '')
    ),
    'Barry Goldwater Scholarship': (
        (data['Gender'] == 1) & 
        (data['Income (Yearly)'] < 60000) & 
        (data['CGPA'] >= 8)
    ),
    'Merit Scholarship': (
        data['College Name'] == "Symbiosis College"
    )
})

# Train multi-label classifier
multi_model = MultiOutputClassifier(RandomForestClassifier())
multi_model.fit(X, multi_target)

# Function to plot scholarship eligibility distribution
def plot_scholarship_distribution():
    scholarship_columns = ['Reliance Scholarship', 'PM Scholarship', 'Indira Gandhi Scholarship', 
                           'Barry Goldwater Scholarship', 'Merit Scholarship']
    
    eligibility = {scholarship: data[scholarship].sum() for scholarship in scholarship_columns}

    fig, ax = plt.subplots()
    ax.bar(eligibility.keys(), eligibility.values(), color='skyblue')
    ax.set_title('Eligibility Distribution for Scholarships')
    ax.set_ylabel('Number of Eligible Students')
    ax.set_xlabel('Scholarships')

    st.pyplot(fig)

# ---------------------- Streamlit App --------------------------
def main():
    st.title("🎓 Student Scholarship Prediction App")

    # Display scholarship criteria directly on front end
    st.markdown("### 📌 Scholarship Eligibility Criteria")
    st.markdown("""
    - **General Scholarship**: Attendance > 75%, CGPA > 6, SGPA > 6  
    - **Reliance Scholarship**: Attendance > 75%, CGPA > 7, SGPA > 7  
    - **PM Scholarship**: Income < 50,000, CGPA > 6, SGPA > 6, Extracurricular Activity required  
    - **Indira Gandhi Scholarship**: Female, CGPA > 6, Extracurricular Activity required  
    - **Barry Goldwater Scholarship**: Male, Income < 60,000, CGPA ≥ 8  
    - **Merit Scholarship**: Student must be from *Symbiosis College*
    """)

    # Select student name
    student_name = st.selectbox("Select Student Name", options=data['Student Name'].unique())

    student = data[data['Student Name'] == student_name].iloc[0]

    # Extract and display details
    gender_display = {0: 'Female', 1: 'Male'}
    st.subheader(f"📋 Details of {student_name}")
    st.write(f"Age: {student['Age']}")
    st.write(f"Gender: {gender_display.get(student['Gender'], 'Other')}")
    st.write(f"CGPA: {student['CGPA']}")
    st.write(f"SGPA: {student['SGPA']}")
    st.write(f"Attendance: {student['Attendance (%)']}%")
    st.write(f"Course: {student['Course']}")
    st.write(f"Branch: {student['Branch']}")
    st.write(f"Extracurricular: {student['Extracurricular']}")
    st.write(f"Additional Activity: {student['Additional Activity']}")
    st.write(f"Income (Yearly): {student['Income (Yearly)']}")
    st.write(f"College: {student['College Name']}")

    if st.button("Predict Scholarship Eligibility"):
        results = []

        # Conditions
        conds = {
            "General Scholarship": student['Attendance (%)'] > 75 and student['CGPA'] > 6 and student['SGPA'] > 6,
            "Reliance Scholarship": student['CGPA'] > 7 and student['SGPA'] > 7 and student['Attendance (%)'] > 75,
            "PM Scholarship": student['Income (Yearly)'] < 50000 and student['CGPA'] > 6 and student['SGPA'] > 6 and student['Extracurricular'],
            "Indira Gandhi Scholarship": student['Gender'] == 0 and student['CGPA'] > 6 and student['Extracurricular'],
            "Barry Goldwater Scholarship": student['Gender'] == 1 and student['Income (Yearly)'] < 60000 and student['CGPA'] >= 8,
            "Merit Scholarship": student['College Name'] == "Symbiosis College"
        }

        for name, condition in conds.items():
            results.append(f"✅ Eligible for {name}" if condition else f"❌ Not Eligible for {name}")

        st.success("\n".join(results))

        # ------------------ Recommendation Section -------------------
        st.subheader("🎯 Scholarship Recommendations")
        recommended = [name for name, cond in conds.items() if cond]

        if recommended:
            st.markdown("Based on the student's profile, they are best suited for:")
            for i, name in enumerate(recommended):
                st.markdown(f"**{i+1}. {name}**")
        else:
            st.warning("❌ No scholarships match this student's profile. Please improve eligibility criteria.")

    # Show lists of eligible and not eligible students
    scholarships = {
        "Reliance Scholarship": (data['CGPA'] > 7.0) & (data['SGPA'] > 7.0) & (data['Attendance (%)'] > 75),
        "PM Scholarship": (data['Income (Yearly)'] < 50000) & (data['CGPA'] > 6.0) & (data['SGPA'] > 6.0) & (data['Extracurricular'] != ''),
        "Indira Gandhi Scholarship": (data['Gender'] == 0) & (data['CGPA'] > 6.0) & (data['Extracurricular'] != ''),
        "Barry Goldwater Scholarship": (data['Gender'] == 1) & (data['Income (Yearly)'] < 60000) & (data['CGPA'] >= 8.0),
        "Merit Scholarship": (data['College Name'] == "Symbiosis College")
    }

    4

    if st.button("Show All Eligible Students"):
        for name, condition in scholarships.items():
            eligible = data[condition].copy()
            eligible['Gender'] = eligible['Gender'].map(gender_display)
            st.subheader(f"✅ {len(eligible)} Eligible for {name}")
            st.write(eligible[['Student Name', 'Age', 'Gender', 'CGPA', 'SGPA', 'Attendance (%)', 'Course', 'Branch']])

    if st.button("Show All Not Eligible Students"):
        for name, condition in scholarships.items():
            not_eligible = data[~condition].copy()
            not_eligible['Gender'] = not_eligible['Gender'].map(gender_display)
            st.subheader(f"❌ {len(not_eligible)} Not Eligible for {name}")
            st.write(not_eligible[['Student Name', 'Age', 'Gender', 'CGPA', 'SGPA', 'Attendance (%)', 'Course', 'Branch']])

    st.subheader("📊 Overall Summary")
    total_students = len(data)
    total_eligible = sum(data[cond].shape[0] for cond in scholarships.values())
    st.write(f"✅ Total Eligible Students: {total_eligible}")
    st.write(f"❌ Total Not Eligible Students: {total_students - total_eligible}")

if __name__ == "__main__":
    main()
