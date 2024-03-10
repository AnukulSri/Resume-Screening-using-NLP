import streamlit as st 
import re
import pickle
import nltk

clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def clean_cv (txt):
    review = re.sub('http\S+\s',' ',txt)
    review = re.sub('RT|cc', ' ', review)
    review = re.sub('#\S+\s', ' ', review)
    review = re.sub('@\S+', '  ', review)  
    review = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', review)
    review = re.sub(r'[^\x00-\x7f]', ' ', review) 
    review = re.sub('\s+', ' ', review)
    return review

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume',type=['txt','Pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        clean_res = clean_cv(resume_text)
        inp_feature = tfidf.transform([clean_res])
        pred_id = clf.predict(inp_feature)[0]

        st.write(pred_id)

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(pred_id,"unkown")
        st.write("Predicted Category is : ",category_name)

if __name__ == "__main__":
    main()