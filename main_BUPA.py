import os, re
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
import time

# Flask App
app = Flask(__name__)

# Free API tokens (get from Hugging Face)
HF_TOKEN = "your API key"

# Explicitly specify model for local fallback
try:
    from transformers import pipeline
    # Use a smaller, efficient model
    local_qa = pipeline('question-answering', model="distilbert-base-cased-distilled-squad")
except ImportError:
    local_qa = None
except Exception as e:
    print(f"Error loading local model: {str(e)}")
    local_qa = None

general_exclusion_list = ["HIV/AIDS", "Parkinson's disease", "Alzheimer's disease", 
                         "pregnancy", "substance abuse", "self-inflicted injuries", 
                         "sexually transmitted diseases(std)", "pre-existing conditions"]

# Policy document cache
policy_docs = {
    "claim_approval": "Documents required: ID proof, medical bills, doctor's prescription, hospital discharge summary.",
    "exclusions": "General exclusions: " + ", ".join(general_exclusion_list)
}

def get_file_content(file):
    text = ""
    if file.filename.endswith(".pdf"):
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_expense_with_regex(text):
    """Robust regex method for expense extraction"""
    import re
    patterns = [
        # Improved patterns to handle various formats
        r'(?:Total|Grand Total|Amount Due)[^\d]*[\$\₹\€\£]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
        r'[\$\₹\€\£]\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?!\d)',
        r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2}))\b(?=\D*$)'
    ]
    
    amounts = []
    for pattern in patterns:
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean and convert to float
                    clean_amount = match.replace(',', '')
                    amounts.append(float(clean_amount))
                except ValueError:
                    continue
        except Exception as e:
            print(f"Regex error with pattern '{pattern}': {str(e)}")
    
    # Return the largest amount found (likely the total)
    return max(amounts) if amounts else None

def extract_diagnosis_with_regex(text):
    """Robust regex to find diagnosis"""
    import re
    patterns = [
        r'Diagnosis:\s*(.*?)(?:\n|$)',
        r'Condition:\s*(.*?)(?:\n|$)',
        r'Reason for Visit:\s*(.*?)(?:\n|$)',
        r'Treatment for:\s*(.*?)(?:\n|$)',
        r'Presenting Complaint:\s*(.*?)(?:\n|$)',
        r'Chief Complaint:\s*(.*?)(?:\n|$)'
    ]
    
    for pattern in patterns:
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        except:
            continue
    
    # Fallback: Look for medical terms
    medical_terms = r'\b(pregnancy|fever|cancer|diabetes|fracture|injury|infection|bodyache|headache|asthma)\b'
    try:
        match = re.search(medical_terms, text, re.IGNORECASE)
        if match:
            return match.group(0)
    except:
        pass
    
    return None

def clean_and_convert_amount(amount):
    """Convert amount to float, handling strings and currency symbols"""
    if amount is None:
        return 0.0
        
    if isinstance(amount, str):
        # Remove currency symbols and commas
        clean_amount = re.sub(r'[^\d.]', '', amount)
        try:
            return float(clean_amount)
        except ValueError:
            return 0.0
    elif isinstance(amount, (int, float)):
        return float(amount)
    else:
        return 0.0

def get_bill_info(data):
    """Extract disease and expense from medical bill text"""
    # First try to extract with regex
    diagnosis = extract_diagnosis_with_regex(data)
    expense = extract_expense_with_regex(data)
    
    if diagnosis and expense is not None:
        return {'disease': diagnosis, 'expense': expense}
    
    # If regex fails, try Hugging Face API
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    prompt = f"Extract the primary medical condition/disease and total expense from this medical bill: {data} Return JSON format: {{'disease':'','expense':''}}"
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        result = response.json()
        if result and isinstance(result, list) and 'generated_text' in result[0]:
            try:
                return json.loads(result[0]['generated_text'])
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_match = re.search(r'\{.*\}', result[0]['generated_text'])
                if json_match:
                    return json.loads(json_match.group(0))
    except requests.exceptions.RequestException as e:
        print(f"Hugging Face API error: {str(e)}")
    except (json.JSONDecodeError, KeyError):
        print("Failed to parse JSON response")
    
    # Fallback to local model if available
    if local_qa:
        try:
            disease = local_qa(question="What is the primary medical condition or disease being treated?", context=data)['answer']
            expense = local_qa(question="What is the total expense amount on this medical bill?", context=data)['answer']
            return {
                'disease': disease, 
                'expense': clean_and_convert_amount(expense)
            }
        except Exception as e:
            print(f"Local model error: {str(e)}")
    
    # Final fallback to regex results
    return {
        'disease': diagnosis or patient_info.get('claim_reason', 'See Claim Reason'), 
        'expense': expense or extract_expense_with_regex(data)  # Try again if first attempt failed
    }

def is_disease_excluded(disease, exclusion_list):
    """Check if disease matches any exclusion using simple string matching"""
    if not disease:
        return False
        
    # Check with simple string matching
    disease_lower = disease.lower()
    for exclusion in exclusion_list:
        if exclusion.lower() in disease_lower:
            return True
            
    return False

def generate_claim_report(patient_info, bill_info, claim_amount_str):
    """Generate claim report with robust numeric handling"""
    # Convert claim amount to float
    try:
        claim_amount = float(claim_amount_str)
    except ValueError:
        claim_amount = 0.0

    # Get and clean bill expense
    bill_expense = clean_and_convert_amount(bill_info.get('expense'))
    
    # Get disease from bill or fallback to claim reason
    disease = bill_info.get('disease', patient_info.get('claim_reason', ''))
    
    # Basic validation checks
    info_complete = all([
        patient_info.get('name'),
        patient_info.get('address'),
        patient_info.get('claim_reason'),
        bill_expense > 0  # Ensure we have valid expense
    ])
    
    excluded = is_disease_excluded(disease, general_exclusion_list)
    
    # Decision logic
    if not info_complete:
        decision = "REJECTED: Incomplete information"
    elif claim_amount > bill_expense:
        decision = f"REJECTED: Claim amount ({claim_amount}) exceeds bill amount ({bill_expense})"
    elif excluded:
        decision = f"REJECTED: Treatment for '{disease}' is excluded"
    else:
        decision = f"APPROVED: Up to ${bill_expense}"
    
    # Generate report
    report = f"""
    CLAIM DECISION: {decision}
    
    Executive Summary
    This report details the analysis of the insurance claim submitted by {patient_info['name']}. 
    The claim has been evaluated based on completeness of information, policy exclusions, and amount validation.
    
    Claim Details
    - Patient: {patient_info['name']}
    - Claim Reason: {patient_info['claim_reason']}
    - Medical Facility: {patient_info['medical_facility']}
    - Date of Service: {patient_info['date']}
    - Claim Amount: ${claim_amount}
    - Bill Amount: ${bill_expense}
    - Detected Condition: {disease}
    
    Verification Results
    - Information Complete: {'Yes' if info_complete else 'No'}
    - Covered Condition: {'No' if excluded else 'Yes'}
    - Amount Valid: {'No' if claim_amount > bill_expense else 'Yes'}
    
    Final Decision
    {decision}
    """
    return report

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_claim():
    # Extract form data
    patient_info = {
        'name': request.form['name'],
        'address': request.form['address'],
        'claim_type': request.form['claim_type'],
        'claim_reason': request.form['claim_reason'],
        'date': request.form['date'],
        'medical_facility': request.form['medical_facility'],
        'total_claim_amount': request.form['total_claim_amount'],
        'description': request.form['description']
    }
    medical_bill = request.files['medical_bill']
    
    # Process bill
    bill_text = get_file_content(medical_bill)
    if not bill_text.strip():
        return render_template("result.html", 
                              output="The uploaded bill is empty or could not be read.",
                              **patient_info)
    
    bill_info = get_bill_info(bill_text)
    
    # Handle expense extraction failures
    if bill_info.get('expense') is None:
        return render_template("result.html", 
                              output="Could not extract expense amount from bill. Please resubmit with clearer documentation.",
                              **patient_info)
    
    # Generate claim report
    report = generate_claim_report(patient_info, bill_info, patient_info['total_claim_amount'])
    
    # Format for HTML display
    formatted_report = re.sub(r'\n\s+', '<br>', report)
    
    return render_template("result.html", 
                          output=formatted_report,
                          **patient_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
